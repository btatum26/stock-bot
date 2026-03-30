from typing import Dict, Any, List
import pandas as pd
import numpy as np
from ..base import Feature, FeatureResult, register_feature

@register_feature("RSI")
class RSI(Feature):
    @property
    def name(self) -> str:
        return "RSI"

    @property
    def description(self) -> str:
        return "Relative Strength Index (Wilder's Smoothing)."

    @property
    def category(self) -> str:
        return "Oscillators (Momentum)"
    
    @property
    def outputs(self) -> List[str | None]:
        return [None]

    def compute(self, df: pd.DataFrame, params: Dict[str, Any], cache: Any = None) -> FeatureResult:
        period = int(params.get("period", 14))
        ob = float(params.get("overbought", 70))
        os = float(params.get("oversold", 30))
        norm_method = params.get("normalize", "none")
        color = params.get("color", "#aaff00")
        
        close = df['Close'] if 'Close' in df.columns else df['close']
        
        # Calculate price changes
        delta = close.diff()
        
        # Calculate gains and losses using Wilder's Smoothing
        gain = (delta.where(delta > 0, 0.0)).ewm(alpha=1/period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/period, adjust=False).mean()
        
        # Relative Strength (RS) and RSI calculation
        rs = gain / loss
        rsi = np.where(loss == 0, 100, 100 - (100 / (1 + rs)))
        rsi = pd.Series(rsi, index=df.index)
        
        col_name = self.generate_column_name("RSI", params)
        
        # Apply normalization to the bounded RSI series
        final_data = self.normalize(df, rsi, norm_method)
        
        return FeatureResult(data={col_name: final_data})
