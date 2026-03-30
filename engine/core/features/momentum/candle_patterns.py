from typing import Dict, Any, List
import pandas as pd
import numpy as np
from ..base import Feature, FeatureResult, OutputSchema, OutputType, Pane, register_feature


@register_feature("CandlePatterns")
class CandlePatterns(Feature):
    @property
    def name(self) -> str:
        return "Candle Patterns"

    @property
    def description(self) -> str:
        return "Detects classic candlestick patterns (Doji, Hammer)."

    @property
    def category(self) -> str:
        return "Patterns"

    @property
    def output_schema(self) -> List[OutputSchema]:
        return [
            OutputSchema(name="doji", output_type=OutputType.MARKER, pane=Pane.OVERLAY),
            OutputSchema(name="hammer", output_type=OutputType.MARKER, pane=Pane.OVERLAY),
        ]

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "doji_threshold": 0.1,
            "hammer_ratio": 2.0,
        }

    @property
    def parameter_options(self) -> Dict[str, Dict[str, Any]]:
        return {
            "doji_threshold": {"min": 0.01, "max": 0.3, "step": 0.01},
            "hammer_ratio": {"min": 1.0, "max": 5.0, "step": 0.5},
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any], cache: Any = None) -> FeatureResult:
        doji_thresh = float(params.get("doji_threshold", 0.1))
        hammer_ratio = float(params.get("hammer_ratio", 2.0))

        open_p = df['Open'] if 'Open' in df.columns else df['open']
        close_p = df['Close'] if 'Close' in df.columns else df['close']
        high_p = df['High'] if 'High' in df.columns else df['high']
        low_p = df['Low'] if 'Low' in df.columns else df['low']

        body = (close_p - open_p).abs()
        rng = high_p - low_p

        # Doji: small body relative to range
        is_doji = ((body <= (rng * doji_thresh)) & (rng > 0)).astype(int)

        # Hammer: small body at top, long lower wick
        lower_wick = np.minimum(open_p, close_p) - low_p
        upper_wick = high_p - np.maximum(open_p, close_p)
        is_hammer = ((lower_wick >= (body * hammer_ratio)) & (upper_wick <= (body * 0.5))).astype(int)

        # Sparse markers: replace 0 with NaN so only pattern occurrences show
        doji_markers = is_doji.replace(0, np.nan) * close_p
        hammer_markers = is_hammer.replace(0, np.nan) * low_p

        col_doji = self.generate_column_name("CandlePatterns", params, "doji")
        col_hammer = self.generate_column_name("CandlePatterns", params, "hammer")

        return FeatureResult(data={
            col_doji: doji_markers,
            col_hammer: hammer_markers,
        })
