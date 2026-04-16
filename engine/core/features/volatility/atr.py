from typing import Dict, Any, List
import pandas as pd
import numpy as np
from ..base import Feature, FeatureResult, OutputSchema, OutputType, Pane, register_feature

@register_feature("AverageTrueRange")
@register_feature("ATR")
class AverageTrueRange(Feature):
    @property
    def name(self) -> str: 
        return "ATR"

    @property
    def description(self) -> str: 
        return "Average True Range."

    @property
    def category(self) -> str: 
        return "Volatility"

    @property
    def output_schema(self) -> List[OutputSchema]:
        return [
            OutputSchema(name=None, output_type=OutputType.LINE, pane=Pane.NEW),
        ]

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "period": 14,
            "normalize": ["none", "z_score", "pct_distance", "price_ratio"]
        }

    def non_stationary_outputs(self, params: Dict[str, Any]) -> List[str]:
        # Raw ATR scales with price level -> non-stationary. Normalized variants
        # (pct_distance, price_ratio, z_score) are stationary and pass through
        # MinMax untouched.
        if params.get("normalize", "none") != "none":
            return []
        return [self.generate_column_name("AverageTrueRange", params)]

    def compute(self, df: pd.DataFrame, params: Dict[str, Any], cache: Any = None) -> FeatureResult:
        period = int(params.get("period", 14))
        norm_method = params.get("normalize", "none")
        
        high = df['High'] if 'High' in df.columns else df['high']
        low = df['Low'] if 'Low' in df.columns else df['low']
        close = df['Close'] if 'Close' in df.columns else df['close']
        
        close_prev = close.shift(1)

        # Calculate the three components of True Range
        tr1 = high - low
        tr2 = (high - close_prev).abs()
        tr3 = (low - close_prev).abs()

        # True Range is the maximum of the three components
        tr = np.maximum(tr1, np.maximum(tr2, tr3))

        # ATR is the simple moving average of True Range
        atr = tr.rolling(window=period).mean()

        # Apply systematic normalization
        final_data = self.normalize(df, atr, norm_method)

        col_name = self.generate_column_name("AverageTrueRange", params)
        return FeatureResult(data={col_name: final_data})
