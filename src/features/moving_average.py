from typing import Dict, Any, List
import pandas as pd
from .base import Feature, FeatureOutput, LineOutput

class MovingAverage(Feature):
    @property
    def name(self) -> str:
        return "Simple Moving Average"

    @property
    def description(self) -> str:
        return "Standard Trend Indicator."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "period": 50,
            "color": "#ff9900"
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> List[FeatureOutput]:
        period = int(params.get("period", 50))
        color = params.get("color", "#ff9900")
        
        sma = df['Close'].rolling(window=period).mean()
        
        # Replace NaNs with None for JSON serialization or safe plotting
        data_list = sma.where(pd.notnull(sma), None).tolist()
        
        return [
            LineOutput(
                name=f"SMA {period}",
                data=data_list,
                color=color,
                width=2
            )
        ]
