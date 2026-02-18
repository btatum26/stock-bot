from typing import Dict, Any, List
import pandas as pd
import numpy as np
from .base import Feature, FeatureOutput, LineOutput

class AverageTrueRange(Feature):
    @property
    def name(self) -> str:
        return "ATR (Scaled)"

    @property
    def description(self) -> str:
        return "Average True Range scaled to fit chart. Measures volatility."

    @property
    def category(self) -> str:
        return "Volatility"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "period": 14,
            "color": "#ff0000"
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> List[FeatureOutput]:
        period = int(params.get("period", 14))
        
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        # Scale to Price Range
        min_price = low.min()
        max_price = high.max()
        p_range = max_price - min_price
        
        min_atr = atr.min()
        max_atr = atr.max()
        
        if max_atr != min_atr:
            atr_scaled = ((atr - min_atr) / (max_atr - min_atr)) * p_range + min_price
        else:
            atr_scaled = atr
            
        return [
            LineOutput(
                name=f"ATR {period}",
                data=atr_scaled.where(pd.notnull(atr_scaled), None).tolist(),
                color=params.get("color", "#ff0000"),
                width=2
            )
        ]
