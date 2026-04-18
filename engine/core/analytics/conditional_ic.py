"""
Conditional IC surface analysis.

Takes a daily IC series from ICAnalyzer and a list of macro dimensions,
bins the macro variables into quantile-based regime cells, and computes
mean IC and IC-IR per cell.

Diagnosis:
  flat       -- IC is roughly constant across regimes (signal works everywhere)
  structured -- clear high-IC and low-IC regions (build regime weighting)
  noisy      -- no coherent structure (wrong state space, or fragile signal)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .macro_fetcher import MacroSpec


@dataclass
class RegimeBin:
    label: str
    mean_ic: float
    ic_ir: float
    n_obs: int
    # dim_label -> (lo, hi) quantile boundaries
    boundaries: Dict[str, Tuple[float, float]] = field(default_factory=dict)


@dataclass
class ConditionalICResult:
    dims: List[MacroSpec]
    horizon: int
    bins: List[RegimeBin]      # sorted by mean_ic descending
    diagnosis: str             # 'flat' | 'structured' | 'noisy' | 'insufficient_data'
    diagnosis_detail: str
    unconditional_ic: float = float("nan")


class ConditionalIC:
    """
    Compute IC conditioned on macro regime dimensions.

    Parameters
    ----------
    ic_series : pd.Series
        Daily IC values at a single horizon (from ICAnalyzer.ic_series[h]).
    horizon : int
        The horizon the IC series represents (for display only).
    """

    def __init__(self, ic_series: pd.Series, horizon: int = 5):
        self.ic_series = ic_series.dropna()
        self.horizon = horizon

    def compute(
        self,
        macro_dims: List[Tuple[pd.Series, MacroSpec]],
        min_obs_per_bin: int = 20,
    ) -> ConditionalICResult:
        """
        Parameters
        ----------
        macro_dims : list of (fetched_series, MacroSpec)
        min_obs_per_bin : int
            Bins with fewer observations are excluded from the output.
        """
        # Align all macro series to IC trading-day index via forward-fill
        aligned = pd.DataFrame({"ic": self.ic_series})
        for series, spec in macro_dims:
            aligned[spec.label] = series.reindex(self.ic_series.index, method="ffill")
        aligned = aligned.dropna()

        dims = [spec for _, spec in macro_dims]
        unconditional = float(aligned["ic"].mean()) if len(aligned) else float("nan")

        if len(aligned) < 30:
            return ConditionalICResult(
                dims=dims,
                horizon=self.horizon,
                bins=[],
                diagnosis="insufficient_data",
                diagnosis_detail=f"Only {len(aligned)} aligned observations after dropna.",
                unconditional_ic=unconditional,
            )

        # Bin each macro dimension into quantiles; fall back to equal-width on failure
        bin_edges: Dict[str, Tuple[np.ndarray, MacroSpec]] = {}
        for series, spec in macro_dims:
            col = spec.label
            try:
                aligned[f"{col}__bin"], edges = pd.qcut(
                    aligned[col], spec.n_bins, labels=False, retbins=True, duplicates="drop"
                )
            except Exception:
                aligned[f"{col}__bin"], edges = pd.cut(
                    aligned[col], spec.n_bins, labels=False, retbins=True
                )
            bin_edges[col] = (edges, spec)

        group_cols = [f"{spec.label}__bin" for _, spec in macro_dims]
        result_bins: List[RegimeBin] = []

        for group_key, group_df in aligned.groupby(group_cols):
            if not isinstance(group_key, tuple):
                group_key = (group_key,)

            n = len(group_df)
            if n < min_obs_per_bin:
                continue

            ic_vals = group_df["ic"].values
            mu = float(np.nanmean(ic_vals))
            sigma = float(np.nanstd(ic_vals, ddof=1))
            ic_ir = (mu / sigma) if sigma > 0 else float("nan")

            boundaries: Dict[str, Tuple[float, float]] = {}
            label_parts = []
            for (col, (edges, spec)), bin_idx in zip(bin_edges.items(), group_key):
                idx = int(bin_idx)
                lo = float(edges[idx])
                hi = float(edges[idx + 1]) if idx + 1 < len(edges) else float(edges[-1])
                boundaries[spec.label] = (round(lo, 4), round(hi, 4))
                label_parts.append(f"{spec.label}=[{lo:.2f},{hi:.2f}]")

            result_bins.append(RegimeBin(
                label="  ".join(label_parts),
                mean_ic=mu,
                ic_ir=ic_ir if math.isfinite(ic_ir) else float("nan"),
                n_obs=n,
                boundaries=boundaries,
            ))

        result_bins.sort(key=lambda b: b.mean_ic if math.isfinite(b.mean_ic) else -999, reverse=True)
        diagnosis, detail = self._diagnose(result_bins, unconditional)

        return ConditionalICResult(
            dims=dims,
            horizon=self.horizon,
            bins=result_bins,
            diagnosis=diagnosis,
            diagnosis_detail=detail,
            unconditional_ic=unconditional,
        )

    @staticmethod
    def _diagnose(bins: List[RegimeBin], unconditional: float) -> Tuple[str, str]:
        if not bins:
            return "noisy", "No bins met the minimum observation threshold."

        ic_vals = [b.mean_ic for b in bins if math.isfinite(b.mean_ic)]
        if not ic_vals:
            return "noisy", "All bins have degenerate IC values."

        ic_range = max(ic_vals) - min(ic_vals)
        top_ic = ic_vals[0]
        bot_ic = ic_vals[-1]

        if ic_range < 0.02:
            return (
                "flat",
                f"IC range across regimes is {ic_range:.4f} (< 0.02). "
                "Signal behaves uniformly — no regime filter needed.",
            )

        if top_ic > unconditional + 0.02:
            return (
                "structured",
                f"IC spread = {ic_range:.4f}. Best regime IC = {top_ic:.4f}, "
                f"worst = {bot_ic:.4f}, unconditional = {unconditional:.4f}. "
                "Build strategy with regime weighting.",
            )

        return (
            "noisy",
            f"IC range = {ic_range:.4f} but no coherent regime lift above unconditional "
            f"({unconditional:.4f}). Macro state space may be wrong, or signal is fragile.",
        )
