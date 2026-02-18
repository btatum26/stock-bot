from typing import Dict, Any, List
import pandas as pd
import numpy as np
from .base import Feature, FeatureOutput, MarkerOutput

class CandlePatterns(Feature):
    @property
    def name(self) -> str:
        return "Candle Patterns"

    @property
    def description(self) -> str:
        return "Detects classic candlestick patterns (Doji, Hammer)."

    @property
    def category(self) -> str:
        return "Signals & Patterns"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "doji_threshold": 0.1, # Body < 10% of range
            "hammer_ratio": 2.0 # Lower wick > 2x body
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> List[FeatureOutput]:
        doji_thresh = float(params.get("doji_threshold", 0.1))
        hammer_ratio = float(params.get("hammer_ratio", 2.0))
        
        open_p = df['Open']
        close_p = df['Close']
        high_p = df['High']
        low_p = df['Low']
        
        body = abs(close_p - open_p)
        rng = high_p - low_p
        
        # Doji Logic: Small body relative to range
        is_doji = (body <= (rng * doji_thresh)) & (rng > 0)
        
        # Hammer Logic: Small body at top, long lower wick
        lower_wick = np.minimum(open_p, close_p) - low_p
        upper_wick = high_p - np.maximum(open_p, close_p)
        is_hammer = (lower_wick >= (body * hammer_ratio)) & (upper_wick <= (body * 0.5))
        
        doji_indices = np.where(is_doji)[0].tolist()
        doji_prices = df['High'].iloc[doji_indices] * 1.001 # Slightly above
        
        hammer_indices = np.where(is_hammer)[0].tolist()
        hammer_prices = df['Low'].iloc[hammer_indices] * 0.999 # Slightly below
        
        return [
            MarkerOutput(
                name="Doji",
                indices=doji_indices,
                values=doji_prices.tolist(),
                shape='d', # diamond
                color='#ffff00'
            ),
            MarkerOutput(
                name="Hammer",
                indices=hammer_indices,
                values=hammer_prices.tolist(),
                shape='t', # triangle up (t1 in pyqtgraph usually, checking map)
                color='#00ff00'
            )
        ]
