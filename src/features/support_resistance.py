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
            "threshold_pct": 0.015, # For identifying the pivot itself (ZigZag)
            "window": 5, # Only for SG or BW
            "clustering_pct": 0.02, # For merging nearby pivots
            "min_strength": 1.0
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> List[FeatureOutput]:
        method = params.get("method", "ZigZag")
        threshold = float(params.get("threshold_pct", 0.015))
        window = int(params.get("window", 5))
        cluster_thresh = float(params.get("clustering_pct", 0.02))
        min_str = float(params.get("min_strength", 1.0))

        # Pass the clustering threshold to the analyzer
        analyzer = LevelsAnalyzer(threshold_pct=cluster_thresh)
        
        pivots = []
        if method == "Savitzky-Golay":
            pivots = analyzer.get_pivots_smoothed(df, window=window)
        elif method == "Bill Williams":
            pivots = analyzer.get_pivots_bill_williams(df, window=window)
        else: # ZigZag
            pivots = analyzer.get_pivots_zigzag(df, deviation_pct=threshold) 

        clusters = analyzer.cluster_pivots(pivots)
        
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
