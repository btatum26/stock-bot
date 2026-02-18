from typing import Dict, Any, List
import pandas as pd
import numpy as np
from .base import Feature, FeatureOutput, LineOutput

class AnchoredVWAP(Feature):
    @property
    def name(self) -> str:
        return "Anchored VWAP"

    @property
    def description(self) -> str:
        return "VWAP calculation starting from a specific number of bars ago."

    @property
    def category(self) -> str:
        return "Volume & Profile"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "anchor_bars_back": 100, # How many bars back to start
            "color": "#00d8ff"
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> List[FeatureOutput]:
        bars_back = int(params.get("anchor_bars_back", 100))
        color = params.get("color", "#00d8ff")
        
        if bars_back >= len(df):
            start_idx = 0
        else:
            start_idx = len(df) - bars_back
            
        # Calculate Typical Price
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        v_tp = tp * df['Volume']
        
        # Slice from anchor
        v_tp_slice = v_tp.iloc[start_idx:]
        vol_slice = df['Volume'].iloc[start_idx:]
        
        # Cumulative Sum from anchor
        cum_v_tp = v_tp_slice.cumsum()
        cum_vol = vol_slice.cumsum()
        
        vwap_slice = cum_v_tp / cum_vol
        
        # Prepare full-length array with None before anchor
        vwap_full = [None] * start_idx + vwap_slice.tolist()
        
        return [
            LineOutput(
                name=f"AVWAP ({bars_back})",
                data=vwap_full,
                color=color,
                width=2
            )
        ]
