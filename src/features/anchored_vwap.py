from typing import Dict, Any
import pandas as pd
from .base import Feature, FeatureResult

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
            "anchor_bars_back": 100
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> FeatureResult:
        bars_back = int(params.get("anchor_bars_back", 100))
        
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
        
        # Create full series aligned with df.index
        vwap_series = pd.Series(index=df.index, dtype=float)
        vwap_series.iloc[start_idx:] = vwap_slice
        
        return FeatureResult(data={f"AVWAP_{bars_back}": vwap_series})
