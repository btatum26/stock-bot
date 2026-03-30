from typing import Dict, Any, List
import pandas as pd
from ..base import Feature, FeatureResult, OutputSchema, OutputType, Pane, register_feature

@register_feature("AnchoredVWAP")
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
    def output_schema(self) -> List[OutputSchema]:
        return [
            OutputSchema(name=None, output_type=OutputType.LINE, pane=Pane.OVERLAY),
        ]

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "anchor_bars_back": 100,
            "normalize": "none"
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any], cache: Any = None) -> FeatureResult:
        bars_back = int(params.get("anchor_bars_back", 100))
        norm_method = params.get("normalize", "none")
        
        # Determine the starting index for the anchored calculation
        if bars_back >= len(df):
            start_idx = 0
        else:
            start_idx = len(df) - bars_back
            
        high = df['High'] if 'High' in df.columns else df['high']
        low = df['Low'] if 'Low' in df.columns else df['low']
        close = df['Close'] if 'Close' in df.columns else df['close']
        volume = df['Volume'] if 'Volume' in df.columns else df['volume']
            
        # Typical Price calculation
        tp = (high + low + close) / 3
        v_tp = tp * volume
        
        # Calculate cumulative volume and price-volume product from the anchor point
        v_tp_slice = v_tp.iloc[start_idx:]
        vol_slice = volume.iloc[start_idx:]
        
        cum_v_tp = v_tp_slice.cumsum()
        cum_vol = vol_slice.cumsum()
        
        vwap_slice = cum_v_tp / cum_vol
        
        # Map the slice back to a full-length series aligned with the input DataFrame
        vwap_series = pd.Series(index=df.index, dtype=float)
        vwap_series.iloc[start_idx:] = vwap_slice
        
        # Apply normalization for machine learning features
        final_data = self.normalize(df, vwap_series, norm_method)
        
        col_name = self.generate_column_name("AnchoredVWAP", params)
        return FeatureResult(data={col_name: final_data})
