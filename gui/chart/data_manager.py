"""
ChartDataManager — data fetching and per-bar label pre-computation.

No Qt dependencies; pure data logic. ChartWindow calls load() and stores
the returned DataFrame and label arrays on itself.
"""

import logging
import random

import pandas as pd

from engine import ModelEngine
from ..config import DEFAULT_TICKERS


class ChartDataManager:
    """Fetches OHLCV data and pre-computes per-bar label strings for the chart."""

    def __init__(self, engine: ModelEngine):
        self._engine = engine

    def load_random_ticker(self) -> str:
        """Pick a random ticker from the local cache, or fall back to a hardcoded default."""
        try:
            tickers = self._engine.list_cached_tickers()
        except Exception:
            tickers = []
        return random.choice(tickers) if tickers else random.choice(DEFAULT_TICKERS)

    def load(self, ticker: str, interval: str) -> tuple:
        """Fetch and prepare chart data for *ticker* at *interval*.

        Returns:
            (df, labels) where labels is a dict with keys:
              tick_intraday, tick_daily, mouse_ts_labels, mouse_vol_labels
            Returns (None, {}) on any error.
        """
        try:
            df = self._engine.get_historical_data(ticker, interval)
        except Exception:
            logging.exception(f"Error fetching {ticker} ({interval})")
            return None, {}

        if df is None or df.empty:
            return None, {}

        # DataBroker returns lowercase columns; chart overlays expect Title-case
        df.columns = [c.capitalize() for c in df.columns]

        dti = pd.DatetimeIndex(df.index)
        vol_arr = df["Volume"].values
        labels = {
            "tick_intraday":    dti.strftime("%H:%M\n%m-%d").tolist(),
            "tick_daily":       dti.strftime("%Y-%m-%d").tolist(),
            "mouse_ts_labels":  dti.strftime("%Y-%m-%d %H:%M").tolist(),
            "mouse_vol_labels": [
                f"Vol: {v/1e6:.2f}M" if v > 1e6
                else f"Vol: {v/1e3:.1f}K" if v > 1e3
                else f"Vol: {int(v)}"
                for v in vol_arr
            ],
        }
        return df, labels
