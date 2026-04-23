"""Regime orchestrator: assembles macro features and produces a RegimeContext.

Call ``RegimeOrchestrator().build_context(df, detector_name)`` from the
backtester immediately after feature computation.  The orchestrator:

  1. Fetches VIX, VIX3M, SPY, HYG from yfinance and aligns to df's index.
  2. Computes ADX from the strategy's own OHLCV data.
  3. Assembles a macro_features DataFrame.
  4. Instantiates and fits the requested detector.
  5. Runs predict_proba (forward algorithm for HMM, deterministic for rule-based).
  6. Runs BOCPD alongside any detector to produce a calibrated novelty score.
  7. Returns a RegimeContext ready to pass to generate_signals.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler

from .base import REGIME_REGISTRY, RegimeContext
from .bocpd import BayesianCPD

logger = logging.getLogger(__name__)

# External tickers fetched for macro features
_MACRO_TICKERS = ["^VIX", "^VIX3M", "SPY", "HYG"]


class RegimeOrchestrator:
    """Builds macro features and applies a regime detector to produce RegimeContext."""

    def build_context(
        self,
        df: pd.DataFrame,
        detector_name: str = "vix_adx",
    ) -> RegimeContext:
        """Build and return a RegimeContext for the given price DataFrame.

        Args:
            df: Feature-enriched OHLCV DataFrame with DatetimeIndex. OHLCV
                columns are used for ADX computation; the index defines the
                output timeline.
            detector_name: Key into REGIME_REGISTRY (e.g. ``'vix_adx'``,
                ``'term_structure'``, ``'hmm'``).

        Returns:
            RegimeContext with proba, labels, novelty aligned to df.index.
        """
        if detector_name not in REGIME_REGISTRY:
            available = list(REGIME_REGISTRY.keys())
            raise ValueError(
                f"Unknown regime detector '{detector_name}'. "
                f"Available: {available}"
            )

        macro = self._build_macro_features(df)

        detector = REGIME_REGISTRY[detector_name]()
        detector.fit(macro)
        proba = detector.predict_proba(macro)
        detector_novelty = detector.novelty_score(macro)

        # BOCPD novelty always runs alongside (gives proper structural-break signal)
        bocpd_novelty = self._run_bocpd(macro)

        # For HMM use BOCPD novelty; for rule-based use max of both (usually 0)
        if detector_name == "hmm":
            novelty = bocpd_novelty
        else:
            novelty = pd.Series(
                np.maximum(bocpd_novelty.values, detector_novelty.values),
                index=macro.index,
            )

        # Align everything to df.index
        shared_idx = df.index
        proba_aligned   = proba.reindex(shared_idx).ffill().fillna(1.0 / detector.n_states)
        novelty_aligned = novelty.reindex(shared_idx).ffill().fillna(0.0)
        labels_aligned  = pd.Series(
            proba_aligned.values.argmax(axis=1),
            index=shared_idx,
            dtype=int,
        )

        return RegimeContext(
            detector_name=detector_name,
            proba=proba_aligned,
            labels=labels_aligned,
            novelty=novelty_aligned,
            n_states=detector.n_states,
        )

    # ------------------------------------------------------------------
    # Macro feature assembly
    # ------------------------------------------------------------------

    def _build_macro_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fetch external macro data and compute internal indicators."""
        idx = df.index
        start = (idx.min() - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
        end   = (idx.max() + pd.Timedelta(days=2)).strftime("%Y-%m-%d")

        result = pd.DataFrame(index=idx)

        # External market data
        ext = self._fetch_external(start, end, idx)
        result = pd.concat([result, ext], axis=1)

        # ADX from the strategy's own OHLCV
        try:
            result["adx"] = self._compute_adx(df).reindex(idx)
        except Exception as e:
            logger.warning(f"ADX computation failed: {e}")
            result["adx"] = np.nan

        return result

    def _fetch_external(
        self, start: str, end: str, target_idx: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """Download VIX/VIX3M/SPY/HYG and align to target_idx."""
        out = pd.DataFrame(index=target_idx)

        try:
            raw = yf.download(
                _MACRO_TICKERS, start=start, end=end,
                progress=False, auto_adjust=True,
            )
            if raw.empty:
                logger.warning("yfinance returned empty data for macro tickers.")
                return out

            if isinstance(raw.columns, pd.MultiIndex):
                raw = raw["Close"]

            if raw.index.tz is not None:
                raw.index = raw.index.tz_localize(None)

            # Reindex to target: forward-fill across weekends / holidays
            aligned = raw.reindex(
                target_idx.union(raw.index)
            ).sort_index().ffill().reindex(target_idx)

            # VIX level
            if "^VIX" in aligned.columns:
                out["vix"] = aligned["^VIX"]

            # VIX term structure ratio
            if "^VIX" in aligned.columns and "^VIX3M" in aligned.columns:
                out["vix3m"] = aligned["^VIX3M"]
                denom = aligned["^VIX3M"].replace(0.0, np.nan)
                out["vix_vix3m"] = aligned["^VIX"] / denom

            # SPY features for HMM
            if "SPY" in aligned.columns:
                spy = aligned["SPY"]
                out["spy_ret"]  = np.log(spy / spy.shift(1))
                out["spy_rvol"] = out["spy_ret"].rolling(21).std() * np.sqrt(252)

            # High-yield spread proxy (HYG 5-day log-change; rising = spread widening)
            if "HYG" in aligned.columns:
                hyg = aligned["HYG"]
                out["hy_spread_chg"] = -np.log(hyg / hyg.shift(5))  # inverted: price↑ = spread↓

        except Exception as e:
            logger.warning(f"Macro data fetch failed: {e}")

        return out

    def _compute_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Compute ADX from OHLCV columns without touching the feature registry."""
        high  = df.get("high",  df.get("High"))
        low   = df.get("low",   df.get("Low"))
        close = df.get("close", df.get("Close"))

        if high is None or low is None or close is None:
            raise ValueError("OHLCV columns not found in df")

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ], axis=1).max(axis=1)

        dm_pos = (high - high.shift(1)).clip(lower=0.0)
        dm_neg = (low.shift(1) - low).clip(lower=0.0)
        dm_pos = dm_pos.where(dm_pos > dm_neg, 0.0)
        dm_neg = dm_neg.where(dm_neg > dm_pos, 0.0)

        alpha = 1.0 / period
        atr    = tr.ewm(alpha=alpha, adjust=False).mean()
        di_pos = 100 * dm_pos.ewm(alpha=alpha, adjust=False).mean() / atr.replace(0, np.nan)
        di_neg = 100 * dm_neg.ewm(alpha=alpha, adjust=False).mean() / atr.replace(0, np.nan)

        dx  = 100 * (di_pos - di_neg).abs() / (di_pos + di_neg).replace(0, np.nan)
        adx = dx.ewm(alpha=alpha, adjust=False).mean()
        return adx

    # ------------------------------------------------------------------
    # BOCPD novelty scoring
    # ------------------------------------------------------------------

    def _run_bocpd(self, macro: pd.DataFrame) -> pd.Series:
        """Run BOCPD on standardised VIX and return P(run_length < 5)."""
        try:
            vix = macro["vix"].ffill().fillna(20.0)
            # Standardise so the Normal-Gamma prior is meaningful
            scaler = StandardScaler()
            vix_std = scaler.fit_transform(vix.values.reshape(-1, 1)).ravel()
            novelty_arr = BayesianCPD().run(vix_std)
            return pd.Series(novelty_arr, index=macro.index)
        except Exception as e:
            logger.warning(f"BOCPD novelty computation failed: {e}")
            return pd.Series(0.0, index=macro.index)
