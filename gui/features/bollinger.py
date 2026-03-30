from typing import Dict, Any
import pandas as pd
from .base import Feature, FeatureResult

class BollingerBands(Feature):
    @property
    def name(self) -> str:
        return "Bollinger Bands"

    @property
    def description(self) -> str:
        return "Volatility bands placed above and below a moving average."

    @property
    def category(self) -> str:
        return "Volatility"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "period": 20,
            "std_dev": 2.0
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> FeatureResult:
        period = int(params.get("period", 20))
        std_multiplier = float(params.get("std_dev", 2.0))
        
        sma = df['Close'].rolling(window=period).mean()
        std = df['Close'].rolling(window=period).std()
        
        upper = sma + (std * std_multiplier)
        lower = sma - (std * std_multiplier)
        
        return FeatureResult(data={
            f"BB_Upper_{period}_{std_multiplier}": upper,
            f"BB_Lower_{period}_{std_multiplier}": lower,
            f"BB_Mid_{period}": sma
        })
