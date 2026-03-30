from typing import Dict, Any, List
import pandas as pd
import numpy as np
from .base import Feature, FeatureOutput, LineOutput

class OnBalanceVolume(Feature):
    @property
    def name(self) -> str:
        return "On-Balance Volume"

    @property
    def description(self) -> str:
        return "Cumulative volume flow."

    @property
    def category(self) -> str:
        return "Volume & Profile"

    @property
    def target_pane(self) -> str:
        return "new"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "color": "#ffff00"
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> List[FeatureOutput]:
        change = df['Close'].diff()
        direction = np.where(change > 0, 1, -1)
        direction[0] = 0
        direction = np.where(change == 0, 0, direction)
        
        obv = (direction * df['Volume']).cumsum()
        
        data_list = obv.where(pd.notnull(obv), None).tolist()
        
        return [
            LineOutput(
                name="OBV",
                data=data_list,
                color=params.get("color", "#ffff00"),
                width=2
            )
        ]
