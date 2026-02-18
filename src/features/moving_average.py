from typing import Dict, Any, List
import pandas as pd
from .base import Feature, FeatureOutput, LineOutput

class MovingAverage(Feature):
    @property
    def name(self) -> str:
        return "Moving Average"

    @property
    def description(self) -> str:
        return "Trend indicator (SMA, EMA)."

    @property
    def category(self) -> str:
        return "Trend Indicators"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "period": 50,
            "type": ["SMA", "EMA"],
            "color": "#ff9900"
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> List[FeatureOutput]:
        period = int(params.get("period", 50))
        ma_type = params.get("type", "SMA")
        color = params.get("color", "#ff9900")
        
        if ma_type == "EMA":
            ma = df['Close'].ewm(span=period, adjust=False).mean()
        else:
            ma = df['Close'].rolling(window=period).mean()
        
        data_list = ma.where(pd.notnull(ma), None).tolist()
        
        return [
            LineOutput(
                name=f"{ma_type} {period}",
                data=data_list,
                color=color,
                width=2
            )
        ]
