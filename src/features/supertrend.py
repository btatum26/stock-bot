from typing import Dict, Any, List
import pandas as pd
import numpy as np
from .base import Feature, FeatureOutput, LineOutput

class Supertrend(Feature):
    @property
    def name(self) -> str:
        return "Supertrend"

    @property
    def description(self) -> str:
        return "Trend-following indicator using ATR to set stop-loss levels."

    @property
    def category(self) -> str:
        return "Trend Indicators"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "period": 10,
            "multiplier": 3.0,
            "color_up": "#00ff00",
            "color_down": "#ff0000"
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> List[FeatureOutput]:
        period = int(params.get("period", 10))
        multiplier = float(params.get("multiplier", 3.0))
        
        # Calculate ATR
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        # Calculate Basic Bands
        hl2 = (high + low) / 2
        basic_upper = hl2 + (multiplier * atr)
        basic_lower = hl2 - (multiplier * atr)
        
        # Final Bands
        final_upper = [0.0] * len(df)
        final_lower = [0.0] * len(df)
        supertrend = [0.0] * len(df)
        
        # 1 means Uptrend, -1 means Downtrend
        trend = [1] * len(df) 
        
        # Iterative calculation required for Supertrend logic
        for i in range(period, len(df)):
            if basic_upper.iloc[i] < final_upper[i-1] or close.iloc[i-1] > final_upper[i-1]:
                final_upper[i] = basic_upper.iloc[i]
            else:
                final_upper[i] = final_upper[i-1]
                
            if basic_lower.iloc[i] > final_lower[i-1] or close.iloc[i-1] < final_lower[i-1]:
                final_lower[i] = basic_lower.iloc[i]
            else:
                final_lower[i] = final_lower[i-1]
                
            # Trend Switch Logic
            if trend[i-1] == 1: # Was Up
                if close.iloc[i] < final_lower[i-1]: # Break below lower
                    trend[i] = -1
                else:
                    trend[i] = 1
            else: # Was Down
                if close.iloc[i] > final_upper[i-1]: # Break above upper
                    trend[i] = 1
                else:
                    trend[i] = -1
            
            if trend[i] == 1:
                supertrend[i] = final_lower[i]
            else:
                supertrend[i] = final_upper[i]

        # Convert 0.0 to None for initial period
        results = []
        for i in range(len(supertrend)):
            if i < period: supertrend[i] = None
        
        # Split into Green/Red lines for visualization? 
        # Or just one line that changes color? 
        # FeatureOutput defines one color. So we return two lines, one for Up, one for Down.
        
        up_line = [val if t == 1 else None for val, t in zip(supertrend, trend)]
        down_line = [val if t == -1 else None for val, t in zip(supertrend, trend)]
        
        return [
            LineOutput(name="Supertrend Up", data=up_line, color=params.get("color_up"), width=2),
            LineOutput(name="Supertrend Down", data=down_line, color=params.get("color_down"), width=2)
        ]
