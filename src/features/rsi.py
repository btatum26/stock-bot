from typing import Dict, Any, List
import pandas as pd
import numpy as np
from .base import Feature, FeatureOutput, LineOutput

class RSI(Feature):
    @property
    def name(self) -> str:
        return "RSI"

    @property
    def description(self) -> str:
        return "Relative Strength Index (0-100)."

    @property
    def category(self) -> str:
        return "Oscillators (Momentum)"

    @property
    def target_pane(self) -> str:
        return "new"

    @property
    def y_range(self) -> List[float]:
        return [0, 100]

    @property
    def y_padding(self) -> float:
        return 0.05

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> List[FeatureOutput]:
        period = int(params.get("period", 14))
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return [
            LineOutput(
                name=f"RSI {period}",
                data=rsi.where(pd.notnull(rsi), None).tolist(),
                color=params.get("color", "#aaff00"),
                width=2
            )
        ]
