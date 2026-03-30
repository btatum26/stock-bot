from typing import Dict, Any, List
import pandas as pd
import numpy as np
from ..base import Feature, FeatureResult, register_feature

@register_feature("KeltnerChannels")
class KeltnerChannels(Feature):
    @property
    def name(self) -> str:
        return "Keltner Channels"

    @property
    def description(self) -> str:
        return "Volatility channels based on ATR and EMA."

    @property
    def category(self) -> str:
        return "Volatility"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "ema_period": 20,
            "atr_period": 10,
            "multiplier": 2.0,
            "normalize": "none"
        }

    @property
    def outputs(self) -> List[str]:
        return ["upper", "center", "lower"]

    def compute(self, df: pd.DataFrame, params: Dict[str, Any], cache: Any = None) -> FeatureResult:
        ema_period = int(params.get("ema_period", 20))
        atr_period = int(params.get("atr_period", 10))
        multiplier = float(params.get("multiplier", 2.0))
        norm_method = params.get("normalize", "none")
        
        # Calculate Center Line and ATR using Cache
        if cache:
            center_line = cache.get_series("EMA", {"period": ema_period}, df)
            atr = cache.get_series("ATR", {"period": atr_period}, df)
        else:
            close = df['Close'] if 'Close' in df.columns else df['close']
            high = df['High'] if 'High' in df.columns else df['high']
            low = df['Low'] if 'Low' in df.columns else df['low']
            
            center_line = close.ewm(span=ema_period, adjust=False).mean()
            
            close_prev = close.shift(1)
            tr1 = high - low
            tr2 = (high - close_prev).abs()
            tr3 = (low - close_prev).abs()
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            atr = tr.rolling(window=atr_period).mean()
        
        # Upper and Lower bands are offsets of the center line based on ATR
        upper_band = center_line + (multiplier * atr)
        lower_band = center_line - (multiplier * atr)
        
        col_upper = self.generate_column_name("KeltnerChannels", params, "upper")
        col_center = self.generate_column_name("KeltnerChannels", params, "center")
        col_lower = self.generate_column_name("KeltnerChannels", params, "lower")

        # Apply normalization to the center line and bands
        final_center = self.normalize(df, center_line, norm_method)
        final_upper = self.normalize(df, upper_band, norm_method)
        final_lower = self.normalize(df, lower_band, norm_method)
        
        data_dict = {
            col_upper: final_upper,
            col_center: final_center,
            col_lower: final_lower
        }
        
        return FeatureResult(data=data_dict)
