from typing import Dict, Any, List
import pandas as pd
import numpy as np
from .base import Feature, FeatureOutput, LineOutput

class OnBalanceVolume(Feature):
    @property
    def name(self) -> str:
        return "On-Balance Volume (Scaled)"

    @property
    def description(self) -> str:
        return "Cumulative volume flow, scaled to fit the price chart for divergence analysis."

    @property
    def category(self) -> str:
        return "Volume & Profile"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "color": "#ffff00"
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> List[FeatureOutput]:
        # Calculate OBV
        # If Close > PrevClose: +Vol, else -Vol
        change = df['Close'].diff()
        direction = np.where(change > 0, 1, -1)
        direction[0] = 0 # First bar has no diff
        
        # Handle flat days (change == 0) -> 0 volume? 
        # Standard OBV: If close == prev_close, volume is ignored (0 change).
        direction = np.where(change == 0, 0, direction)
        
        obv = (direction * df['Volume']).cumsum()
        
        # Scale to Price Range for Overlay Visualization
        min_obv = obv.min()
        max_obv = obv.max()
        min_price = df['Low'].min()
        max_price = df['High'].max()
        
        if max_obv != min_obv:
            obv_scaled = ((obv - min_obv) / (max_obv - min_obv)) * (max_price - min_price) + min_price
        else:
            obv_scaled = obv # Flat line if no volume change
            
        data_list = obv_scaled.where(pd.notnull(obv_scaled), None).tolist()
        
        return [
            LineOutput(
                name="OBV (Scaled)",
                data=data_list,
                color=params.get("color", "#ffff00"),
                width=2
            )
        ]
