from typing import Dict, Any
import pandas as pd
from .base import Feature, FeatureResult

class AverageTrueRange(Feature):
    @property
    def name(self) -> str:
        return "ATR"

    @property
    def description(self) -> str:
        return "Average True Range. Measures volatility."

    @property
    def category(self) -> str:
        return "Volatility"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "period": 14
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> FeatureResult:
        period = int(params.get("period", 14))
        
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return FeatureResult(data={f"ATR_{period}": atr})
