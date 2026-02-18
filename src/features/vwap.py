from typing import Dict, Any, List
import pandas as pd
import numpy as np
from .base import Feature, FeatureOutput, LineOutput

class VWAP(Feature):
    @property
    def name(self) -> str:
        return "VWAP"

    @property
    def description(self) -> str:
        return "Volume Weighted Average Price (Anchored to start of data)."

    @property
    def category(self) -> str:
        return "Volume & Profile"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "color": "#00d8ff"
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> List[FeatureOutput]:
        # Typical Price
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        # Volume * TP
        v_tp = tp * df['Volume']
        
        # Cumulative Sums (Anchored to start of loaded data)
        # For true daily VWAP on intraday, we'd need to reset every day.
        # Here we implement a simple rolling or anchored VWAP for the view.
        # Let's do Anchored (Cumulative) as it's most common for trend analysis.
        
        cum_v_tp = v_tp.cumsum()
        cum_v = df['Volume'].cumsum()
        
        vwap = cum_v_tp / cum_v
        
        return [
            LineOutput(
                name="VWAP",
                data=vwap.where(pd.notnull(vwap), None).tolist(),
                color=params.get("color"),
                width=2
            )
        ]
