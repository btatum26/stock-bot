"""Combinatorial Purged Cross-Validation (CPCV) splitter.

Implements the cross-validation methodology from Marcos Lopez de Prado's
*Advances in Financial Machine Learning*. Standard k-fold CV is invalid
for time-series data because it ignores temporal dependencies. CPCV
addresses this by:

1. Dividing the dataset into N equal groups.
2. Generating all C(N, k) combinations where k groups form the test set.
3. Applying a **purge** protocol to remove training samples that could
   leak information into the test set (within ``l_max`` bars before
   each test block).
4. Applying an **embargo** protocol to remove training samples
   immediately after each test block (a percentage-based buffer).
"""

import itertools
import numpy as np
import pandas as pd
from typing import List, Tuple

from ..logger import logger


class CPCVSplitter:
    """Generates combinatorial purged cross-validation folds.

    Each fold uses a unique combination of k groups as the test set and
    the remaining N-k groups as the training set. Purge and embargo
    protocols remove training observations near test-set boundaries to
    prevent information leakage from overlapping feature lookback windows
    and autocorrelated targets.

    Attributes:
        n: Number of groups to divide the dataset into.
        k: Number of groups to allocate to each test fold.
        embargo_pct: Fraction of total dataset length to embargo after
            each test block boundary.
    """

    def __init__(
        self,
        n_groups: int = 6,
        k_test_groups: int = 2,
        embargo_pct: float = 0.01,
    ):
        """Initializes the CPCV splitter.

        Args:
            n_groups: Number of contiguous groups to partition the data
                into. Higher values produce more folds but smaller test
                sets. Must be >= 2.
            k_test_groups: Number of groups to use as the test set in
                each fold. Must be >= 1 and < ``n_groups``.
            embargo_pct: Fraction of the total dataset length to remove
                from the training set immediately following each test
                block. Prevents leakage from autocorrelated targets.
                Defaults to 0.01 (1%).

        Raises:
            ValueError: If group parameters are invalid.
        """
        if n_groups < 2:
            raise ValueError(f"n_groups must be >= 2, got {n_groups}")
        if k_test_groups < 1 or k_test_groups >= n_groups:
            raise ValueError(
                f"k_test_groups must be in [1, {n_groups - 1}], "
                f"got {k_test_groups}"
            )
        self.n = n_groups
        self.k = k_test_groups
        self.embargo_pct = embargo_pct

    def split(
        self, df: pd.DataFrame, l_max: int = 0
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generates all CPCV train/test folds with purge and embargo.

        Args:
            df: The feature DataFrame to split. Only its length is used;
                the actual content is not accessed.
            l_max: Maximum feature lookback window. Training observations
                within ``l_max`` positions before any test block are
                purged to prevent feature leakage.

        Returns:
            List of ``(train_idx, test_idx)`` tuples. Each element is a
            numpy array of positional indices. The number of folds
            equals C(N, k).
        """
        n_samples = len(df)
        indices = np.arange(n_samples)

        # Partition into N contiguous groups of roughly equal size
        groups = np.array_split(indices, self.n)

        n_folds = len(list(itertools.combinations(range(self.n), self.k)))
        logger.info(
            f"CPCV: {self.n} groups, k={self.k} -> {n_folds} folds "
            f"(purge={l_max}, embargo={self.embargo_pct:.1%})"
        )

        folds: List[Tuple[np.ndarray, np.ndarray]] = []
        for test_group_ids in itertools.combinations(range(self.n), self.k):
            test_idx = np.concatenate(
                [groups[g] for g in test_group_ids]
            )
            train_group_ids = [
                g for g in range(self.n) if g not in test_group_ids
            ]
            train_idx = np.concatenate(
                [groups[g] for g in train_group_ids]
            )

            # Apply purge protocol
            if l_max > 0:
                train_idx = self._apply_purge(train_idx, test_idx, l_max)

            # Apply embargo protocol
            if self.embargo_pct > 0:
                train_idx = self._apply_embargo(
                    train_idx, test_idx, n_samples
                )

            folds.append((train_idx, test_idx))

        return folds

    # ------------------------------------------------------------------
    # Purge and embargo protocols
    # ------------------------------------------------------------------

    def _apply_purge(
        self,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        l_max: int,
    ) -> np.ndarray:
        """Removes training samples within ``l_max`` rows before each test block.

        This prevents features computed with rolling windows from leaking
        test-period information into training samples. For example, if a
        200-period SMA is used, the 200 training rows immediately before
        a test block contain partial test-period data in their feature
        values.

        Args:
            train_idx: Positional indices of the current training set.
            test_idx: Positional indices of the current test set.
            l_max: Number of rows to purge before each test block start.

        Returns:
            Filtered training indices with purged rows removed.
        """
        blocks = self._contiguous_blocks(test_idx)
        purge_mask = np.zeros(len(train_idx), dtype=bool)
        for block_start, _ in blocks:
            purge_mask |= (train_idx >= block_start - l_max) & (
                train_idx < block_start
            )
        n_purged = purge_mask.sum()
        if n_purged > 0:
            logger.debug(f"Purge removed {n_purged} training samples.")
        return train_idx[~purge_mask]

    def _apply_embargo(
        self,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        n_samples: int,
    ) -> np.ndarray:
        """Removes training samples immediately after each test block.

        This buffer zone prevents leakage from autocorrelated targets.
        If the target is a forward-looking return, training samples just
        after a test block may overlap with the test block's target
        horizon.

        Args:
            train_idx: Positional indices of the current training set
                (already purged).
            test_idx: Positional indices of the current test set.
            n_samples: Total number of samples in the dataset.

        Returns:
            Filtered training indices with embargoed rows removed.
        """
        embargo_size = max(1, int(n_samples * self.embargo_pct))
        blocks = self._contiguous_blocks(test_idx)
        embargo_mask = np.zeros(len(train_idx), dtype=bool)
        for _, block_end in blocks:
            embargo_mask |= (train_idx > block_end) & (
                train_idx <= block_end + embargo_size
            )
        n_embargoed = embargo_mask.sum()
        if n_embargoed > 0:
            logger.debug(f"Embargo removed {n_embargoed} training samples.")
        return train_idx[~embargo_mask]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _contiguous_blocks(indices: np.ndarray) -> List[Tuple[int, int]]:
        """Finds contiguous blocks in a sorted positional index array.

        Args:
            indices: Array of positional indices (need not be sorted on
                entry; sorted internally).

        Returns:
            List of ``(start, end)`` tuples (inclusive) identifying each
            contiguous run of indices.
        """
        if len(indices) == 0:
            return []
        sorted_idx = np.sort(indices)
        breaks = np.where(np.diff(sorted_idx) > 1)[0]
        starts = np.concatenate([[0], breaks + 1])
        ends = np.concatenate([breaks, [len(sorted_idx) - 1]])
        return [
            (int(sorted_idx[s]), int(sorted_idx[e]))
            for s, e in zip(starts, ends)
        ]
