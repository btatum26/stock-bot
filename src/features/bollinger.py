from typing import Dict, Any, List
import pandas as pd
from .base import Feature, FeatureOutput, LineOutput

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
            "std_dev": 2.0,
            "color_upper": "#00ff00",
            "color_lower": "#ff0000",
            "color_mid": "#ffffff"
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> List[FeatureOutput]:
        period = int(params.get("period", 20))
        std_multiplier = float(params.get("std_dev", 2.0))
        
        # Calculate
        sma = df['Close'].rolling(window=period).mean()
        std = df['Close'].rolling(window=period).std()
        
        upper = sma + (std * std_multiplier)
        lower = sma - (std * std_multiplier)
        
        # Helper to clean NaNs
        def clean(series):
            return series.where(pd.notnull(series), None).tolist()

        return [
            LineOutput(name="Upper Band", data=clean(upper), color=params.get("color_upper"), width=1),
            LineOutput(name="Lower Band", data=clean(lower), color=params.get("color_lower"), width=1),
            LineOutput(name="Mid Band", data=clean(sma), color=params.get("color_mid"), width=1)
        ]
