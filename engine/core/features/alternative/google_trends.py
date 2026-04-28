"""Google Trends search-interest feature.

Uses pytrends (rate-limited, ~1 req/sec). Data is weekly from Google Trends
and forward-filled onto the daily price index.

Install: uv add pytrends

Parameters
----------
ticker : str  (required)
    Search keyword (typically the ticker symbol or company name).
median_window : int
    Number of weeks to use for the baseline rolling median (default 8).

Outputs
-------
ratio  : search interest divided by its rolling median (>1 = spike)
zscore : 252-day z-score of the ratio
"""

import logging
import time
from typing import Dict, Any, List
import pandas as pd
from ..base import Feature, FeatureResult, OutputSchema, OutputType, Pane, register_feature

logger = logging.getLogger("model-engine.features.alternative.google_trends")

# In-process cache keyed by (ticker, start, end)
_trends_cache: Dict[str, pd.Series] = {}


def _fetch_trends_raw(ticker: str, start: str, end: str) -> pd.Series:
    """Fetches weekly Google Trends interest for ticker. Returns Series with DatetimeIndex."""
    try:
        from pytrends.request import TrendReq
    except ImportError:
        logger.error("pytrends not installed. Run: uv add pytrends")
        return pd.Series(dtype=float)

    cache_key = f"{ticker}|{start}|{end}"
    if cache_key in _trends_cache:
        return _trends_cache[cache_key]

    try:
        pt = TrendReq(hl="en-US", tz=360, timeout=(10, 25), retries=2, backoff_factor=0.5)
        pt.build_payload([ticker], cat=0, timeframe=f"{start} {end}", geo="US", gprop="")
        time.sleep(1.2)   # stay under Google's rate limit
        df = pt.interest_over_time()

        if df.empty or ticker not in df.columns:
            return pd.Series(dtype=float)

        series = df[ticker].astype(float)
        if series.index.tz is not None:
            series.index = series.index.tz_localize(None)
        series.index = pd.to_datetime(series.index)

        _trends_cache[cache_key] = series
        return series

    except Exception as e:
        logger.warning(f"GoogleTrends fetch failed for {ticker}: {e}")
        return pd.Series(dtype=float)


@register_feature("GoogleTrends")
class GoogleTrends(Feature):
    """Google Search Trends interest ratio for a ticker symbol.

    Spikes in search interest often precede or accompany price moves (especially
    for retail-heavy names). The 8-week median normalises for secular trend changes
    in a ticker's search popularity.

    Rate-limited and best suited for weekly-refreshed universes.
    """

    @property
    def name(self) -> str:
        return "Google Trends"

    @property
    def description(self) -> str:
        return "Search interest ratio vs 8-week median and z-score. Requires pytrends."

    @property
    def category(self) -> str:
        return "Alternative"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "ticker":        "",
            "median_window": 8,   # weeks
        }

    @property
    def output_schema(self) -> List[OutputSchema]:
        return [
            OutputSchema(name=None,      output_type=OutputType.LINE, pane=Pane.NEW),
            OutputSchema(name="zscore",  output_type=OutputType.LINE, pane=Pane.NEW),
        ]

    def compute(self, df: pd.DataFrame, params: Dict[str, Any], cache: Any = None) -> FeatureResult:
        ticker        = params.get("ticker", "")
        median_window = int(params.get("median_window", 8))

        col_ratio  = self.generate_column_name("GoogleTrends", params)
        col_zscore = self.generate_column_name("GoogleTrends", params, "zscore")
        nan        = pd.Series(float("nan"), index=df.index)

        if not ticker:
            logger.warning("GoogleTrends: 'ticker' param is required")
            return FeatureResult(data={col_ratio: nan, col_zscore: nan})

        start = df.index.min().strftime("%Y-%m-%d")
        end   = df.index.max().strftime("%Y-%m-%d")

        raw = _fetch_trends_raw(ticker, start, end)
        if raw.empty:
            return FeatureResult(data={col_ratio: nan, col_zscore: nan})

        # Forward-fill weekly data onto the daily price index
        aligned = raw.reindex(df.index.union(raw.index)).ffill().reindex(df.index)

        # Rolling median uses calendar days; median_window weeks ≈ window * 7 trading days
        min_periods = max(1, median_window)
        roll_median = aligned.rolling(window=median_window * 7, min_periods=min_periods).median()
        ratio = aligned / roll_median.replace(0, float("nan"))

        roll_mean = ratio.rolling(252, min_periods=52).mean()
        roll_std  = ratio.rolling(252, min_periods=52).std()
        zscore    = (ratio - roll_mean) / roll_std.replace(0, float("nan"))

        return FeatureResult(data={col_ratio: ratio, col_zscore: zscore})
