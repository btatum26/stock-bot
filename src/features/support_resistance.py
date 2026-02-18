from typing import Dict, Any, List
import pandas as pd
from .base import Feature, FeatureOutput, LevelOutput
from ..analysis import LevelsAnalyzer

class SupportResistance(Feature):
    @property
    def name(self) -> str:
        return "Support & Resistance"

    @property
    def description(self) -> str:
        return "Identifies key price clusters using Local Extrema, Fractals, or ZigZag."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "method": ["ZigZag", "Savitzky-Golay", "Bill Williams"], 
            "threshold_pct": 0.015, # Filter small pivots
            "window": 5, 
            "clustering_pct": 0.02, # Merge distance
            "min_strength": 1.0,
            "recency": 1.0 # 0=None, 1=Linear, 5=Strong
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> List[FeatureOutput]:
        method = params.get("method", "ZigZag")
        threshold = float(params.get("threshold_pct", 0.015))
        window = int(params.get("window", 5))
        cluster_thresh = float(params.get("clustering_pct", 0.02))
        min_str = float(params.get("min_strength", 1.0))
        recency = float(params.get("recency", 1.0))

        analyzer = LevelsAnalyzer(threshold_pct=cluster_thresh)
        
        pivots = []
        if method == "Savitzky-Golay":
            pivots = analyzer.get_pivots_smoothed(df, window=window, min_dist_pct=threshold)
        elif method == "Bill Williams":
            pivots = analyzer.get_pivots_bill_williams(df, window=window, min_dist_pct=threshold)
        else: # ZigZag
            pivots = analyzer.get_pivots_zigzag(df, deviation_pct=threshold) 

        clusters = analyzer.cluster_pivots(pivots, total_bars=len(df), recency_factor=recency)
        
        outputs = []
        for c in clusters:
            if c['strength'] >= min_str:
                # Map strength to opacity/color logic in the GUI later, 
                # but for now we just return the raw level data.
                outputs.append(LevelOutput(
                    name=f"Level {c['price']}",
                    price=c['price'],
                    min_price=c['min_price'],
                    max_price=c['max_price'],
                    strength=c['strength'],
                    color='#0000ff' # Blue base
                ))
                
        return outputs
