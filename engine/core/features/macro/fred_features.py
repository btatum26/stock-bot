import logging
from typing import Dict, Any, List, Optional
import pandas as pd
from ..base import Feature, FeatureResult, OutputSchema, OutputType, Pane, register_feature

logger = logging.getLogger(__name__)

_fetcher_instance = None


def _get_fetcher():
    global _fetcher_instance
    if _fetcher_instance is None:
        from ...data_broker.fetcher import DataFetcher
        _fetcher_instance = DataFetcher()
    return _fetcher_instance


class FredFeature(Feature):
    """Base for FRED macro series features.

    Subclasses declare SERIES_ID (the FRED series code), LABEL, and optionally
    COLUMN_PREFIX (the registry key used for column naming; defaults to SERIES_ID).

    Each instance produces three outputs:
      - level   : raw FRED value, declared non-stationary → routed through FFD
      - roc5    : 5-day rate of change (stationary)
      - zscore  : 252-day rolling z-score (stationary)
    """

    SERIES_ID: str = ""
    LABEL: str = ""
    COLUMN_PREFIX: str = ""   # leave blank to use SERIES_ID

    @property
    def _col_prefix(self) -> str:
        return self.COLUMN_PREFIX or self.SERIES_ID

    @property
    def name(self) -> str:
        return self.LABEL

    @property
    def description(self) -> str:
        return (
            f"FRED {self.SERIES_ID}: {self.LABEL}. "
            "Outputs raw level (non-stationary), 5-day ROC, and 252-day z-score."
        )

    @property
    def category(self) -> str:
        return "Macro"

    @property
    def output_schema(self) -> List[OutputSchema]:
        return [
            OutputSchema(name="level",  output_type=OutputType.LINE, pane=Pane.NEW),
            OutputSchema(name="roc5",   output_type=OutputType.LINE, pane=Pane.NEW),
            OutputSchema(name="zscore", output_type=OutputType.LINE, pane=Pane.NEW),
        ]

    def non_stationary_outputs(self, params: Dict[str, Any]) -> List[str]:
        return [self.generate_column_name(self._col_prefix, params, "level")]

    def compute(self, df: pd.DataFrame, params: Dict[str, Any], cache: Any = None) -> FeatureResult:
        col_level  = self.generate_column_name(self._col_prefix, params, "level")
        col_roc5   = self.generate_column_name(self._col_prefix, params, "roc5")
        col_zscore = self.generate_column_name(self._col_prefix, params, "zscore")

        nan = pd.Series(float("nan"), index=df.index)

        if df.empty:
            return FeatureResult(data={col_level: nan, col_roc5: nan, col_zscore: nan})

        start = df.index.min().strftime("%Y-%m-%d")
        end   = df.index.max().strftime("%Y-%m-%d")

        try:
            raw = _get_fetcher().fetch_macro_data(self.SERIES_ID, start, end)
        except Exception as e:
            logger.warning(f"FRED fetch failed for {self.SERIES_ID}: {e}")
            return FeatureResult(data={col_level: nan, col_roc5: nan, col_zscore: nan})

        if raw.empty:
            return FeatureResult(data={col_level: nan, col_roc5: nan, col_zscore: nan})

        series = raw.set_index("date")["value"].sort_index()
        series.index = pd.to_datetime(series.index)

        # Forward-fill weekly/sparse FRED releases onto the daily price index
        series = series.reindex(df.index.union(series.index)).ffill().reindex(df.index)

        roc5 = series.pct_change(5)

        roll_mean = series.rolling(252, min_periods=63).mean()
        roll_std  = series.rolling(252, min_periods=63).std()
        zscore    = (series - roll_mean) / roll_std.replace(0, float("nan"))

        return FeatureResult(data={
            col_level:  series,
            col_roc5:   roc5,
            col_zscore: zscore,
        })


@register_feature("NFCI")
class NFCI(FredFeature):
    SERIES_ID = "NFCI"
    LABEL = "Chicago Fed Financial Conditions"


@register_feature("ANFCI")
class ANFCI(FredFeature):
    SERIES_ID = "ANFCI"
    LABEL = "Adjusted Financial Conditions"


@register_feature("HYSpread")
class HYSpread(FredFeature):
    SERIES_ID = "BAMLH0A0HYM2"
    COLUMN_PREFIX = "HYSpread"
    LABEL = "High-Yield Credit Spread"


@register_feature("T10Y2Y")
class T10Y2Y(FredFeature):
    SERIES_ID = "T10Y2Y"
    LABEL = "10Y-2Y Treasury Spread"


@register_feature("T10Y3M")
class T10Y3M(FredFeature):
    SERIES_ID = "T10Y3M"
    LABEL = "10Y-3M Treasury Spread"


@register_feature("VIXCLS")
class VIXCLS(FredFeature):
    SERIES_ID = "VIXCLS"
    LABEL = "VIX (FRED daily)"


@register_feature("ICSA")
class ICSA(FredFeature):
    SERIES_ID = "ICSA"
    LABEL = "Initial Jobless Claims"


@register_feature("DFF")
class DFF(FredFeature):
    SERIES_ID = "DFF"
    LABEL = "Effective Fed Funds Rate"
