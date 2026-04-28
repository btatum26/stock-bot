import logging
from typing import Dict, Any, List
import pandas as pd
import yfinance as yf
from ..base import Feature, FeatureResult, OutputSchema, OutputType, Pane, register_feature

logger = logging.getLogger("model-engine.features.macro.vix_term_structure")


@register_feature("VIXTermStructure")
class VIXTermStructure(Feature):
    """VIX / VIX3M ratio — contango vs backwardation regime signal.

    Below 0.90: deep contango (calm, risk-on).
    Above 1.00: backwardation (stress, risk-off).

    Outputs: raw ratio and 252-day rolling z-score.
    Fetches ^VIX and ^VIX3M from yfinance, aligned to the price df's date range.
    """

    @property
    def name(self) -> str:
        return "VIX Term Structure"

    @property
    def description(self) -> str:
        return "VIX/VIX3M ratio. Contango (<0.90) vs backwardation (>1.0) regime indicator."

    @property
    def category(self) -> str:
        return "Macro"

    @property
    def output_schema(self) -> List[OutputSchema]:
        return [
            OutputSchema(name=None,      output_type=OutputType.LINE, pane=Pane.NEW),
            OutputSchema(name="zscore",  output_type=OutputType.LINE, pane=Pane.NEW),
        ]

    def compute(self, df: pd.DataFrame, params: Dict[str, Any], cache: Any = None) -> FeatureResult:
        col_ratio  = self.generate_column_name("VIXTermStructure", params)
        col_zscore = self.generate_column_name("VIXTermStructure", params, "zscore")
        nan = pd.Series(float("nan"), index=df.index)

        if df.empty:
            return FeatureResult(data={col_ratio: nan, col_zscore: nan})

        start = df.index.min().strftime("%Y-%m-%d")
        end   = df.index.max().strftime("%Y-%m-%d")

        try:
            raw = yf.download(["^VIX", "^VIX3M"], start=start, end=end,
                               progress=False, auto_adjust=True)
            if raw.empty:
                raise ValueError("empty response")

            if isinstance(raw.columns, pd.MultiIndex):
                vix   = raw["Close"]["^VIX"].dropna()
                vix3m = raw["Close"]["^VIX3M"].dropna()
            else:
                raise ValueError("unexpected column format from yfinance multi-ticker download")

            # Ensure tz-naive index
            for s in (vix, vix3m):
                if s.index.tz is not None:
                    s.index = s.index.tz_localize(None)

            ratio = (vix / vix3m).reindex(df.index.union(vix.index)).ffill().reindex(df.index)

        except Exception as e:
            logger.warning(f"VIXTermStructure fetch failed: {e}")
            return FeatureResult(data={col_ratio: nan, col_zscore: nan})

        roll_mean = ratio.rolling(252, min_periods=63).mean()
        roll_std  = ratio.rolling(252, min_periods=63).std()
        zscore    = (ratio - roll_mean) / roll_std.replace(0, float("nan"))

        return FeatureResult(data={col_ratio: ratio, col_zscore: zscore})
