from typing import Dict, Any, List
import pandas as pd
import numpy as np
from ..base import Feature, FeatureResult, OutputSchema, OutputType, Pane, register_feature

@register_feature("OBV")
class OBV(Feature):
    @property
    def name(self) -> str:
        return "On-Balance Volume (OBV)"

    @property
    def description(self) -> str:
        return "Cumulative volume indicator mapping buying/selling pressure."

    @property
    def category(self) -> str:
        return "Volume & Profile"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "normalize": "none"
        }

    @property
    def output_schema(self) -> List[OutputSchema]:
        return [
            OutputSchema(name=None,     output_type=OutputType.LINE, pane=Pane.NEW),
            OutputSchema(name="sma_20", output_type=OutputType.LINE, pane=Pane.NEW),
        ]

    def compute(self, df: pd.DataFrame, params: Dict[str, Any], cache: Any = None) -> FeatureResult:
        norm_method = params.get("normalize", "none")
        
        close = df['Close'] if 'Close' in df.columns else df['close']
        volume = df['Volume'] if 'Volume' in df.columns else df['volume']
        
        # Determine price direction: 1 for up, -1 for down, 0 for flat
        price_diff = close.diff().fillna(0)
        direction = np.sign(price_diff)
        
        # Multiply volume by direction and calculate the cumulative sum
        obv = (volume * direction).cumsum()
        
        # Apply systematic normalization (e.g., Z-Score for unbounded volume indicators)
        final_obv = self.normalize(df, obv, norm_method)
        
        # Calculate a rolling average of OBV for trend filtering
        obv_sma = obv.rolling(window=20).mean()
        final_obv_sma = self.normalize(df, obv_sma, norm_method)
        
        col_name = self.generate_column_name("OBV", params)
        col_sma = self.generate_column_name("OBV", params, "sma_20")
        
        # Return primary and smoothed OBV data
        data_dict = {
            col_name: final_obv,
            col_sma: final_obv_sma
        }
        
        return FeatureResult(data=data_dict)
