from typing import Dict, Any, List
import pandas as pd
import numpy as np
from .base import Feature, FeatureOutput, HeatmapOutput

class VolumeProfile(Feature):
    @property
    def name(self) -> str:
        return "Volume Profile (VPVR)"

    @property
    def description(self) -> str:
        return "Horizontal volume histogram showing high-volume price nodes."

    @property
    def category(self) -> str:
        return "Volume & Profile"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "bins": 100,
            "lookback": 0 # 0 = All visible data
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> List[FeatureOutput]:
        bins = int(params.get("bins", 100))
        lookback = int(params.get("lookback", 0))
        
        if lookback > 0 and len(df) > lookback:
            subset = df.iloc[-lookback:]
        else:
            subset = df
            
        # Create Price Bins
        min_price = subset['Low'].min()
        max_price = subset['High'].max()
        
        # We need to distribute volume into these bins
        # A simple way is to assign each bar's volume to its 'Close' price bin
        # A more accurate way is to distribute volume across High-Low range, 
        # but Close is standard for fast VPVR.
        
        hist, bin_edges = np.histogram(subset['Close'], bins=bins, range=(min_price, max_price), weights=subset['Volume'])
        
        # Normalize histogram for heatmap (0.0 to 1.0)
        if hist.max() > 0:
            hist_norm = hist / hist.max()
        else:
            hist_norm = hist
            
        # Create price grid (center of bins)
        price_grid = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        return [
            HeatmapOutput(
                name="Volume Profile",
                price_grid=price_grid.tolist(),
                density=hist_norm.tolist(),
                color_map="viridis" # Different color than KDE to distinguish
            )
        ]
