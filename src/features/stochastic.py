from typing import Dict, Any, List
import pandas as pd
from .base import Feature, FeatureOutput, LineOutput

class Stochastic(Feature):
    @property
    def name(self) -> str:
        return "Stochastic"

    @property
    def description(self) -> str:
        return "Stochastic Oscillator (%K and %D)."

    @property
    def category(self) -> str:
        return "Oscillators (Momentum)"

    @property
    def target_pane(self) -> str:
        return "new"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "k_period": 14,
            "d_period": 3,
            "color_k": "#00ffff",
            "color_d": "#ff00ff"
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> List[FeatureOutput]:
        k_period = int(params.get("k_period", 14))
        d_period = int(params.get("d_period", 3))
        
        low_min = df['Low'].rolling(window=k_period).min()
        high_max = df['High'].rolling(window=k_period).max()
        
        # %K
        k_percent = 100 * ((df['Close'] - low_min) / (high_max - low_min))
        
        # %D
        d_percent = k_percent.rolling(window=d_period).mean()
        
        def clean(s): return s.where(pd.notnull(s), None).tolist()
        
        return [
            LineOutput(name="%K", data=clean(k_percent), color=params.get("color_k"), width=1),
            LineOutput(name="%D", data=clean(d_percent), color=params.get("color_d"), width=1)
        ]
