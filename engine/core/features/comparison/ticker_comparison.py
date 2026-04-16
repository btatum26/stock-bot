from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from ..base import Feature, FeatureResult, OutputSchema, OutputType, Pane, register_feature

# Module-level cache: avoids re-fetching the same comparison ticker
# across multiple assets in a batch run.  Keyed by (ticker, interval).
_comp_cache: Dict[tuple, pd.DataFrame] = {}


def _infer_interval(index: pd.DatetimeIndex) -> str:
    """Infer the data interval from the DataFrame's datetime index."""
    if len(index) < 2:
        return "1d"
    median_delta = pd.Series(index).diff().dropna().median()
    hours = median_delta.total_seconds() / 3600
    if hours < 1:
        minutes = int(round(median_delta.total_seconds() / 60))
        return f"{minutes}m"
    elif hours <= 1.5:
        return "1h"
    elif hours <= 5:
        return "4h"
    elif hours <= 30:
        return "1d"
    else:
        return "1wk"


@register_feature("TICKER_COMPARE")
class TickerComparison(Feature):
    @property
    def name(self) -> str:
        return "Ticker Comparison"

    @property
    def description(self) -> str:
        return (
            "Compares the current ticker against a reference ticker. "
            "Outputs relative strength ratio, rolling correlation, and rolling beta."
        )

    @property
    def category(self) -> str:
        return "Comparison"

    @property
    def output_schema(self) -> List[OutputSchema]:
        return [
            OutputSchema(name="ratio", output_type=OutputType.LINE, pane=Pane.NEW),
            OutputSchema(name="corr", output_type=OutputType.LINE, pane=Pane.NEW, y_range=(-1, 1)),
            OutputSchema(name="beta", output_type=OutputType.LINE, pane=Pane.NEW),
            OutputSchema(name="regime", output_type=OutputType.LINE, pane=Pane.NEW, y_range=(0, 1)),
        ]

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "compare_ticker": "SPY",
            "window": 20,
        }

    @property
    def parameter_options(self) -> Dict[str, Dict[str, Any]]:
        return {
            "compare_ticker": {"type": "str", "description": "Ticker symbol to compare against"},
            "window": {"type": "int", "min": 5, "max": 252, "description": "Rolling window for correlation and beta"},
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any], cache: Optional[Any] = None) -> FeatureResult:
        compare_ticker = str(params.get("compare_ticker", "SPY")).upper()
        window = int(params.get("window", 20))

        close_col = "Close" if "Close" in df.columns else "close"
        primary_close = df[close_col]

        # Fetch comparison ticker data via the DataBroker (cached across batch calls)
        from engine.core.data_broker import DataBroker
        interval = _infer_interval(df.index)
        cache_key = (compare_ticker, interval)
        if cache_key in _comp_cache:
            comp_df = _comp_cache[cache_key]
        else:
            broker = DataBroker()
            comp_df = broker.get_data(compare_ticker, interval)
            _comp_cache[cache_key] = comp_df

        if comp_df.empty:
            # Return NaN series if comparison data unavailable
            nan_series = pd.Series(np.nan, index=df.index)
            ratio_col = self.generate_column_name("TICKER_COMPARE", params, "ratio")
            corr_col = self.generate_column_name("TICKER_COMPARE", params, "corr")
            beta_col = self.generate_column_name("TICKER_COMPARE", params, "beta")
            regime_col = self.generate_column_name("TICKER_COMPARE", params, "regime")
            return FeatureResult(data={ratio_col: nan_series, corr_col: nan_series, beta_col: nan_series, regime_col: nan_series})

        comp_close_col = "Close" if "Close" in comp_df.columns else "close"
        comp_close = comp_df[comp_close_col]

        # Align on shared dates
        primary_aligned, comp_aligned = primary_close.align(comp_close, join="inner")

        # 1. Relative strength ratio (normalized to start at 1.0)
        ratio = (primary_aligned / primary_aligned.iloc[0]) / (comp_aligned / comp_aligned.iloc[0])

        # 2. Rolling correlation of returns
        primary_ret = primary_aligned.pct_change()
        comp_ret = comp_aligned.pct_change()
        corr = primary_ret.rolling(window=window).corr(comp_ret)

        # 3. Rolling beta = cov(r_primary, r_comp) / var(r_comp)
        rolling_cov = primary_ret.rolling(window=window).cov(comp_ret)
        rolling_var = comp_ret.rolling(window=window).var()
        beta = rolling_cov / rolling_var.replace(0, np.nan)

        # 4. Regime: comparison ticker close vs its SMA(window)
        comp_sma = comp_aligned.rolling(window=window).mean()
        regime = (comp_aligned > comp_sma).astype(float)

        # Reindex back to original df index (forward-fill for any missing bars)
        ratio = ratio.reindex(df.index, method="ffill")
        corr = corr.reindex(df.index, method="ffill")
        beta = beta.reindex(df.index, method="ffill")
        regime = regime.reindex(df.index, method="ffill")

        ratio_col = self.generate_column_name("TICKER_COMPARE", params, "ratio")
        corr_col = self.generate_column_name("TICKER_COMPARE", params, "corr")
        beta_col = self.generate_column_name("TICKER_COMPARE", params, "beta")
        regime_col = self.generate_column_name("TICKER_COMPARE", params, "regime")

        return FeatureResult(data={ratio_col: ratio, corr_col: corr, beta_col: beta, regime_col: regime})
