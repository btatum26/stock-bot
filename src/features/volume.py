from typing import Dict, Any, List
import pandas as pd
from .base import Feature, FeatureResult, LineOutput

class Volume(Feature):
    @property
    def name(self) -> str:
        return "Volume"

    @property
    def description(self) -> str:
        return "Trading volume for each bar."

    @property
    def category(self) -> str:
        return "Volume & Profile"

    @property
    def target_pane(self) -> str:
        return "new"

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> FeatureResult:
        # Just return raw volume as data, and maybe a scaled version for visualization
        volume = df['Volume']
        
        # Visualization: Scale to 0-100 or something reasonable for a separate pane
        v_min = volume.min()
        v_max = volume.max()
        if v_max != v_min:
            v_scaled = (volume - v_min) / (v_max - v_min) * 100
        else:
            v_scaled = volume
            
        visuals = [
            LineOutput(
                name="Volume",
                data=v_scaled.where(pd.notnull(v_scaled), None).tolist(),
                color="#888888",
                width=1
            )
        ]
        
        return FeatureResult(visuals=visuals, data={"Volume": volume})
