"""
Macro data fetcher: FRED (via existing DataFetcher) and yfinance.

Returns pd.Series indexed by date, forward-filled to daily frequency.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import yfinance as yf

from ..data_broker.fetcher import DataFetcher


@dataclass
class MacroSpec:
    series_id: str
    source: str       # 'yf' or 'fred'
    n_bins: int = 4
    label: Optional[str] = None

    def __post_init__(self):
        if self.label is None:
            self.label = self.series_id
        if self.source not in ("yf", "fred"):
            raise ValueError(f"source must be 'yf' or 'fred', got {self.source!r}")


def parse_macro_spec(s: str) -> MacroSpec:
    """
    Parse a CLI macro dimension string into a MacroSpec.

    Format: 'SERIES_ID:SOURCE[:N_BINS]'

    Examples
    --------
    '^VIX:yf:5'       ->  MacroSpec('^VIX', 'yf', 5)
    'T10Y2Y:fred:3'   ->  MacroSpec('T10Y2Y', 'fred', 3)
    '^TNX:yf'         ->  MacroSpec('^TNX', 'yf', 4)   # default 4 bins
    """
    parts = s.strip().split(":")
    if len(parts) < 2:
        raise ValueError(
            f"Invalid macro spec {s!r}. Expected 'SERIES_ID:SOURCE' or 'SERIES_ID:SOURCE:N_BINS'.\n"
            "Examples: '^VIX:yf:5'  'T10Y2Y:fred:3'  '^TNX:yf'"
        )
    series_id = parts[0]
    source = parts[1].lower()
    n_bins = int(parts[2]) if len(parts) >= 3 else 4
    return MacroSpec(series_id=series_id, source=source, n_bins=n_bins)


class MacroFetcher:
    """Fetches and normalises macro series to daily-frequency pd.Series."""

    def __init__(self):
        self._fetcher = DataFetcher()

    def fetch(self, spec: MacroSpec, start: str, end: str) -> pd.Series:
        """Return a daily-frequency pd.Series for the given MacroSpec."""
        if spec.source == "fred":
            return self._fetch_fred(spec.series_id, start, end)
        return self._fetch_yf(spec.series_id, start, end)

    def _fetch_fred(self, series_id: str, start: str, end: str) -> pd.Series:
        df = self._fetcher.fetch_macro_data(series_id, start, end)
        if df.empty:
            return pd.Series(dtype=float, name=series_id)
        s = df.set_index("date")["value"].rename(series_id)
        s.index = pd.to_datetime(s.index)
        daily_idx = pd.date_range(s.index.min(), s.index.max(), freq="D")
        return s.reindex(daily_idx).ffill().rename(series_id)

    def _fetch_yf(self, series_id: str, start: str, end: str) -> pd.Series:
        df = yf.download(series_id, start=start, end=end, progress=False)
        if df.empty:
            return pd.Series(dtype=float, name=series_id)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        close = df["Close"].squeeze()
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.rename(series_id)
        close.index = pd.to_datetime(close.index)
        if close.index.tz is not None:
            close.index = close.index.tz_localize(None)
        daily_idx = pd.date_range(close.index.min(), close.index.max(), freq="D")
        return close.reindex(daily_idx).ffill().rename(series_id)
