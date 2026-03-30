from typing import Dict, Any, List, Optional
import pandas as pd
from ..base import Feature, FeatureResult, OutputSchema, OutputType, Pane, register_feature

@register_feature("MovingAverage")
class MovingAverage(Feature):
    @property
    def name(self) -> str:
        return "Moving Average"

    @property
    def description(self) -> str:
        return "Trend indicator (SMA, EMA)."

    @property
    def category(self) -> str:
        return "Trend Indicators"

    @property
    def output_schema(self) -> List[OutputSchema]:
        return [
            OutputSchema(name=None, output_type=OutputType.LINE, pane=Pane.OVERLAY),
        ]

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "period": 50,
            "type": "SMA",
            "normalize": "none",
        }

    @property
    def parameter_options(self) -> Dict[str, List[Any]]:
        return {
            "type": ["SMA", "EMA"]
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any], cache: Any = None) -> FeatureResult:
        period = int(params.get("period", 50))
        ma_type = params.get("type", "SMA")
        norm_method = params.get("normalize", "none")
        
        close = df['Close'] if 'Close' in df.columns else df['close']
        
        # Calculate the requested type of Moving Average
        if ma_type == "EMA":
            ma = close.ewm(span=period, adjust=False).mean()
        else:
            ma = close.rolling(window=period).mean()
        
        # Determine the feature ID for naming
        # This ensures consistency with how the Alpha Engine registers these classes
        feat_id = "MovingAverage"
        if ma_type == "SMA" and self.__class__.__name__ == "SMA":
            feat_id = "SMA"
        elif ma_type == "EMA" and self.__class__.__name__ == "EMA":
            feat_id = "EMA"
            
        col_name = self.generate_column_name(feat_id, params)
        
        # Apply systematic normalization
        final_data = self.normalize(df, ma, norm_method)
        
        return FeatureResult(data={col_name: final_data})

@register_feature("EMA")
class EMA(MovingAverage):
    @property
    def name(self) -> str:
        return "Exponential Moving Average"

    def compute(self, df: pd.DataFrame, params: Dict[str, Any], cache: Any = None) -> FeatureResult:
        params["type"] = "EMA"
        return super().compute(df, params, cache)

@register_feature("SMA")
class SMA(MovingAverage):
    @property
    def name(self) -> str:
        return "Simple Moving Average"

    def compute(self, df: pd.DataFrame, params: Dict[str, Any], cache: Any = None) -> FeatureResult:
        params["type"] = "SMA"
        return super().compute(df, params, cache)
