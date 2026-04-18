"""
Information Coefficient (IC) pipeline.

Computes cross-sectional rank IC between a candidate signal and forward
returns across a universe of tickers at multiple holding horizons.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


_DEFAULT_HORIZONS = [1, 5, 10, 20, 60]
_HALF_LIFE_PROBE_HORIZONS = [1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 60]


@dataclass
class ICResult:
    signal_name: str
    universe: List[str]
    period_start: str
    period_end: str
    n_trials: int

    mean_ic: Dict[int, float] = field(default_factory=dict)
    ic_ir: Dict[int, float] = field(default_factory=dict)
    ic_pos_frac: Dict[int, float] = field(default_factory=dict)

    # Raw daily IC series per horizon — needed for conditional IC
    ic_series: Dict[int, pd.Series] = field(default_factory=dict)

    # Quintile returns: DataFrame indexed by quintile (1..5), columns = horizons
    quintile_returns: Optional[pd.DataFrame] = None
    quintile_monotonic: Dict[int, bool] = field(default_factory=dict)
    quintile_spread: Dict[int, float] = field(default_factory=dict)

    daily_turnover: float = float("nan")
    half_life_days: float = float("nan")

    # Robustness: ratio of mean IC in first vs second half of period
    period_ic_consistency: float = float("nan")

    @property
    def passes_gate(self) -> bool:
        """Gate: IC >= 0.02 and IC-IR >= 0.3 at 5-day horizon."""
        ic5 = self.mean_ic.get(5, 0.0)
        ir5 = self.ic_ir.get(5, 0.0)
        return math.isfinite(ic5) and math.isfinite(ir5) and ic5 >= 0.02 and ir5 >= 0.3


class ICAnalyzer:
    """
    Cross-sectional IC analysis for a candidate signal.

    Parameters
    ----------
    signals : dict[str, pd.Series]
        Per-ticker signal series. Values in [-1, 1], DatetimeIndex.
    prices  : dict[str, pd.DataFrame]
        Per-ticker OHLCV DataFrames. Must have a 'close' column and
        either a DatetimeIndex or a 'timestamp' column.
    signal_name : str
        Display name for the report header.
    n_trials : int
        Total backtest/train trial count (for the DSR line in the report).
    """

    def __init__(
        self,
        signals: Dict[str, pd.Series],
        prices: Dict[str, pd.DataFrame],
        signal_name: str = "signal",
        n_trials: int = 1,
    ):
        self.signal_name = signal_name
        self.n_trials = n_trials

        self._sig_matrix = pd.DataFrame(signals).sort_index()
        self._close_matrix = self._build_close_matrix(prices)

    def _build_close_matrix(self, prices: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        frames = {}
        for ticker, df in prices.items():
            df = df.copy()
            if not isinstance(df.index, pd.DatetimeIndex):
                if "timestamp" in df.columns:
                    df = df.set_index("timestamp")
                else:
                    continue
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            frames[ticker] = df["close"]
        return pd.DataFrame(frames).sort_index()

    def _fwd_returns(self, horizon: int) -> pd.DataFrame:
        return self._close_matrix.pct_change(horizon).shift(-horizon)

    def _daily_ic(self, horizon: int, min_tickers: int = 5) -> pd.Series:
        """Spearman rank IC at each date across the cross-section."""
        fwd = self._fwd_returns(horizon)
        common_idx = self._sig_matrix.index.intersection(fwd.index)
        sig = self._sig_matrix.loc[common_idx]
        fwd = fwd.loc[common_idx]

        ic_vals = []
        for date in common_idx:
            s_row = sig.loc[date].dropna()
            f_row = fwd.loc[date].dropna()
            both = s_row.index.intersection(f_row.index)
            if len(both) < min_tickers:
                ic_vals.append(float("nan"))
                continue
            corr, _ = spearmanr(s_row[both].values, f_row[both].values)
            ic_vals.append(float(corr) if math.isfinite(corr) else float("nan"))

        return pd.Series(ic_vals, index=common_idx, name=f"IC_{horizon}d")

    def _quintile_returns(self, horizons: List[int]) -> Tuple[pd.DataFrame, Dict[int, float]]:
        """
        Mean annualized forward return by signal quintile for each horizon.
        Returns (quintile_df, spread_dict).
        """
        records = []
        spreads = {}
        for h in horizons:
            fwd = self._fwd_returns(h)
            common_idx = self._sig_matrix.index.intersection(fwd.index)
            sig = self._sig_matrix.loc[common_idx]
            fwd = fwd.loc[common_idx]

            sig_vals, fwd_vals = [], []
            for date in common_idx:
                s_row = sig.loc[date].dropna()
                f_row = fwd.loc[date].dropna()
                both = s_row.index.intersection(f_row.index)
                if len(both) < 5:
                    continue
                sig_vals.extend(s_row[both].values.tolist())
                fwd_vals.extend(f_row[both].values.tolist())

            if not sig_vals:
                records.append({"horizon": h, **{q: float("nan") for q in range(1, 6)}})
                spreads[h] = float("nan")
                continue

            pair_df = pd.DataFrame({"signal": sig_vals, "fwd": fwd_vals}).dropna()
            pair_df["quintile"] = pd.qcut(pair_df["signal"], 5, labels=False) + 1
            q_means = pair_df.groupby("quintile")["fwd"].mean() * (252 / h)
            row = {"horizon": h}
            for q in range(1, 6):
                row[q] = float(q_means.get(q, float("nan")))
            records.append(row)

            q5 = row.get(5, float("nan"))
            q1 = row.get(1, float("nan"))
            spreads[h] = (q5 - q1) if (math.isfinite(q5) and math.isfinite(q1)) else float("nan")

        df = pd.DataFrame(records).set_index("horizon")
        return df, spreads

    def _turnover(self) -> float:
        """Mean daily |signal change| / 2, averaged across tickers. Range [0, 1]."""
        diffs = self._sig_matrix.diff().abs()
        return float(diffs.mean().mean() / 2.0)

    def _half_life(self) -> float:
        """Horizon at which mean IC first falls to half its peak value."""
        ic_at_h: Dict[int, float] = {}
        for h in _HALF_LIFE_PROBE_HORIZONS:
            series = self._daily_ic(h)
            mu = float(series.mean())
            if math.isfinite(mu):
                ic_at_h[h] = mu

        if not ic_at_h:
            return float("nan")

        peak_h = max(ic_at_h, key=lambda h: ic_at_h[h])
        peak_ic = ic_at_h[peak_h]

        if peak_ic <= 0:
            return float("nan")

        half = peak_ic / 2.0
        for h in sorted(ic_at_h):
            if h > peak_h and ic_at_h[h] <= half:
                return float(h)

        return float(max(_HALF_LIFE_PROBE_HORIZONS))

    def _period_consistency(self, ic5: pd.Series) -> float:
        """
        Ratio of mean IC in first half vs second half of period.
        1.0 = identical, lower = less consistent across time.
        """
        clean = ic5.dropna()
        if len(clean) < 40:
            return float("nan")
        half = len(clean) // 2
        m1 = float(clean.iloc[:half].mean())
        m2 = float(clean.iloc[half:].mean())
        if not (math.isfinite(m1) and math.isfinite(m2)):
            return float("nan")
        denom = max(abs(m1), abs(m2))
        if denom == 0:
            return float("nan")
        return float(min(abs(m1), abs(m2)) / denom)

    def run(self, horizons: List[int] = None) -> ICResult:
        horizons = horizons or _DEFAULT_HORIZONS

        tickers = list(self._sig_matrix.columns)
        idx = self._sig_matrix.index
        period_start = str(idx.min().date()) if len(idx) else ""
        period_end = str(idx.max().date()) if len(idx) else ""

        result = ICResult(
            signal_name=self.signal_name,
            universe=tickers,
            period_start=period_start,
            period_end=period_end,
            n_trials=self.n_trials,
        )

        for h in horizons:
            series = self._daily_ic(h)
            clean = series.dropna()
            result.ic_series[h] = series
            if len(clean) < 10:
                result.mean_ic[h] = float("nan")
                result.ic_ir[h] = float("nan")
                result.ic_pos_frac[h] = float("nan")
                continue
            mu = float(clean.mean())
            sigma = float(clean.std(ddof=1))
            result.mean_ic[h] = mu
            result.ic_ir[h] = mu / sigma if sigma > 0 else float("nan")
            result.ic_pos_frac[h] = float((clean > 0).mean())

        q_df, spreads = self._quintile_returns(horizons)
        result.quintile_returns = q_df
        result.quintile_spread = spreads
        for h in horizons:
            if h in q_df.index:
                vals = [q_df.loc[h, q] for q in range(1, 6)
                        if math.isfinite(q_df.loc[h, q])]
                result.quintile_monotonic[h] = (
                    len(vals) == 5 and all(vals[i] <= vals[i + 1] for i in range(4))
                )

        result.daily_turnover = self._turnover()
        result.half_life_days = self._half_life()
        ic5 = result.ic_series.get(5, pd.Series(dtype=float))
        result.period_ic_consistency = self._period_consistency(ic5)

        return result
