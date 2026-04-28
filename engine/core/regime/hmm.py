"""Gaussian HMM regime detector with causal forward-algorithm inference.

Uses hmmlearn for parameter estimation, then runs the forward algorithm
manually so regime probabilities at time t depend only on x_{1:t} (no
lookahead into the future sequence, as Viterbi decoding would introduce).

Monthly label consistency is maintained by matching HMM states across
retrainings via minimum Euclidean distance of emission means.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.preprocessing import StandardScaler

from .base import RegimeDetector, register_regime

logger = logging.getLogger("model-engine.core.regime.hmm")

try:
    from hmmlearn.hmm import GaussianHMM as _GaussianHMM
    _HMMLEARN_AVAILABLE = True
except ImportError:
    _HMMLEARN_AVAILABLE = False
    logger.warning("hmmlearn not installed — GaussianHMMRegime unavailable. "
                   "Run: uv add hmmlearn")


def _match_states(old_means: np.ndarray, new_means: np.ndarray) -> np.ndarray:
    """Return a permutation mapping new states → old states by closest emission mean.

    Greedy nearest-neighbour: picks the closest pair, removes both from
    consideration, repeats. O(n_states^2) — fine for n_states ≤ 5.

    Args:
        old_means: (n_states, n_features) emission means from previous fit.
        new_means: (n_states, n_features) emission means from current fit.

    Returns:
        perm: (n_states,) int array. perm[new_idx] = old_idx.
    """
    n = len(old_means)
    dists = np.linalg.norm(
        old_means[:, None, :] - new_means[None, :, :], axis=-1
    )  # (n_old, n_new)
    perm = np.full(n, -1, dtype=int)
    used_old = set()
    used_new = set()
    for _ in range(n):
        # Mask already assigned rows/cols
        masked = dists.copy()
        for i in used_old:
            masked[i, :] = np.inf
        for j in used_new:
            masked[:, j] = np.inf
        i, j = np.unravel_index(masked.argmin(), masked.shape)
        perm[j] = i
        used_old.add(i)
        used_new.add(j)
    return perm


@register_regime("hmm")
class GaussianHMMRegime(RegimeDetector):
    """HMM-based regime detector.

    Fits a GaussianHMM on four macro features (SPY log-returns, 21-day
    realised volatility, 5-day change in HY-spread proxy, VIX level), then
    runs the *forward algorithm* — not Viterbi — to produce causal posterior
    probabilities P(state_t | x_{1:t}).

    Exponential smoothing (halflife 4 days) is applied to the raw forward
    probabilities to suppress noisy inter-day flipping.

    Required macro_features columns:
        ``spy_ret``, ``spy_rvol``, ``hy_spread_chg``, ``vix``
    """

    _FEATURE_COLS = ["spy_ret", "spy_rvol", "hy_spread_chg", "vix"]

    def __init__(self, n_states: int = 3, smoothing_halflife: int = 4):
        if not _HMMLEARN_AVAILABLE:
            raise RuntimeError(
                "hmmlearn is required for GaussianHMMRegime. "
                "Install with: uv add hmmlearn  (inside engine/)"
            )
        self._n_states = n_states
        self._halflife = smoothing_halflife
        self._model: Optional[_GaussianHMM] = None
        self._scaler = StandardScaler()
        self._fitted = False
        self._prev_means: Optional[np.ndarray] = None

    @property
    def name(self) -> str:
        return "GaussianHMMRegime"

    @property
    def n_states(self) -> int:
        return self._n_states

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, macro_features: pd.DataFrame) -> None:
        """Fit the HMM on the full macro feature history.

        Applies label consistency by matching new emission means to the
        previous fit (when retraining monthly on a rolling window).
        """
        X = self._prepare(macro_features, fit_scaler=True)
        model = _GaussianHMM(
            n_components=self._n_states,
            covariance_type="full",
            n_iter=200,
            tol=1e-4,
            random_state=42,
        )
        model.fit(X)

        # Align state labels with previous fit to prevent label flipping
        if self._prev_means is not None:
            try:
                perm = _match_states(self._prev_means, model.means_)
                model.means_ = model.means_[perm]
                model.covars_ = model.covars_[perm]
                model.startprob_ = model.startprob_[perm]
                model.transmat_ = model.transmat_[perm][:, perm]
            except Exception as e:
                logger.debug(f"State re-labelling failed: {e}")

        self._prev_means = model.means_.copy()
        self._model = model
        self._fitted = True
        logger.info(
            f"HMM fitted: {self._n_states} states, "
            f"log-likelihood={model.score(X):.2f}"
        )

    def predict_proba(self, macro_features: pd.DataFrame) -> pd.DataFrame:
        """Run the forward algorithm and return smoothed state posteriors."""
        if not self._fitted:
            raise RuntimeError("Call fit() before predict_proba()")
        X = self._prepare(macro_features, fit_scaler=False)
        alpha = self._forward(X)  # (T, n_states), causal

        df = pd.DataFrame(alpha, index=macro_features.index,
                          columns=list(range(self._n_states)))

        # Exponential smoothing to dampen noisy daily flipping
        for col in df.columns:
            df[col] = df[col].ewm(halflife=self._halflife).mean()

        # Re-normalise after smoothing (ewm can shift probabilities off-simplex)
        row_sums = df.sum(axis=1).replace(0, 1.0)
        df = df.div(row_sums, axis=0)
        return df

    def novelty_score(self, macro_features: pd.DataFrame) -> pd.Series:
        """Per-bar negative log-likelihood normalised to [0, 1].

        High values mean the current observation is unlikely under the fitted
        emission distributions — a signal that we may be in a novel regime.
        """
        if not self._fitted:
            return pd.Series(0.0, index=macro_features.index)

        X = self._prepare(macro_features, fit_scaler=False)
        log_e = self._log_emission(X)       # (T, n_states)
        alpha = self._forward(X)            # (T, n_states)
        log_alpha = np.log(np.maximum(alpha, 1e-300))

        # Per-step log-likelihood: log sum_k alpha_k * emission_k
        ll = logsumexp(log_alpha + log_e, axis=1)   # (T,)

        novelty = pd.Series(-ll, index=macro_features.index)
        lo, hi = novelty.min(), novelty.max()
        if hi > lo:
            novelty = (novelty - lo) / (hi - lo)
        else:
            novelty = pd.Series(0.0, index=macro_features.index)
        return novelty

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare(self, macro_features: pd.DataFrame, fit_scaler: bool) -> np.ndarray:
        missing = [c for c in self._FEATURE_COLS if c not in macro_features.columns]
        if missing:
            raise ValueError(f"GaussianHMMRegime: missing macro columns {missing}")
        X = macro_features[self._FEATURE_COLS].ffill().fillna(0.0).values
        return self._scaler.fit_transform(X) if fit_scaler else self._scaler.transform(X)

    def _log_emission(self, X: np.ndarray) -> np.ndarray:
        """log P(x_t | state=k) for all t and k.  Shape: (T, n_states)."""
        T = len(X)
        log_p = np.zeros((T, self._n_states))
        for k in range(self._n_states):
            try:
                log_p[:, k] = multivariate_normal.logpdf(
                    X,
                    mean=self._model.means_[k],
                    cov=self._model.covars_[k],
                )
            except Exception:
                log_p[:, k] = -500.0
        return log_p

    def _forward(self, X: np.ndarray) -> np.ndarray:
        """Causal forward pass returning P(state=k | x_{1:t}) for each t.

        The forward algorithm (not Viterbi) ensures that the posterior at
        bar t depends only on observations up to and including bar t.
        """
        T = len(X)
        n = self._n_states
        transmat = self._model.transmat_          # (n, n)
        startprob = self._model.startprob_        # (n,)
        emission_p = np.exp(self._log_emission(X))  # (T, n)

        alpha = np.zeros((T, n))

        # t = 0: initialise with start probabilities
        alpha[0] = startprob * emission_p[0]
        s = alpha[0].sum()
        alpha[0] = alpha[0] / s if s > 1e-300 else np.full(n, 1.0 / n)

        # t = 1..T-1: predict → update → normalise
        for t in range(1, T):
            predicted = transmat.T @ alpha[t - 1]      # (n,) marginal over prev state
            alpha[t] = predicted * emission_p[t]
            s = alpha[t].sum()
            alpha[t] = alpha[t] / s if s > 1e-300 else np.full(n, 1.0 / n)

        return alpha
