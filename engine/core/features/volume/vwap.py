from typing import Dict, Any
import pandas as pd
from ..base import Feature, FeatureResult, register_feature

@register_feature("VWAP")
class VWAP(Feature):
    @property
    def name(self) -> str: 
        return "VWAP"

    @property
    def description(self) -> str: 
        return "Volume Weighted Average Price."

    @property
    def category(self) -> str: 
        return "Volume & Profile"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "normalize": "none"
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any], cache: Any = None) -> FeatureResult:
        norm_method = params.get("normalize", "none")
        
        high = df['High'] if 'High' in df.columns else df['high']
        low = df['Low'] if 'Low' in df.columns else df['low']
        close = df['Close'] if 'Close' in df.columns else df['close']
        volume = df['Volume'] if 'Volume' in df.columns else df['volume']
        
        dates = df.index.date
        tp = (high + low + close) / 3
        v_tp = tp * volume
        
        cum_v_tp = v_tp.groupby(dates).cumsum()
        cum_v = volume.groupby(dates).cumsum()
        vwap = cum_v_tp / cum_v
        
        # Apply systematic normalization
        final_data = self.normalize(df, vwap, norm_method)
        
        col_name = self.generate_column_name("VWAP", params)
        return FeatureResult(data={col_name: final_data})
