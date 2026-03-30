from typing import Dict, Any
import pandas as pd
import numpy as np
from .base import Feature, FeatureResult

class CCI(Feature):
    @property
    def name(self) -> str:
        return "CCI"

    @property
    def description(self) -> str:
        return "Commodity Channel Index."

    @property
    def category(self) -> str:
        return "Oscillators (Momentum)"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "period": 20
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> FeatureResult:
        period = int(params.get("period", 20))
        
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        sma = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        
        cci = (tp - sma) / (0.015 * mad)
        
        return FeatureResult(data={f"CCI_{period}": cci})
