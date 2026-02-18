from typing import Dict, Any, List
import pandas as pd
import numpy as np
from .base import Feature, FeatureOutput, LineOutput

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
    def target_pane(self) -> str:
        return "new"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "period": 20,
            "color": "#ff00aa"
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> List[FeatureOutput]:
        period = int(params.get("period", 20))
        
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        sma = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        
        cci = (tp - sma) / (0.015 * mad)
        
        data_list = cci.where(pd.notnull(cci), None).tolist()
        
        return [
            LineOutput(
                name=f"CCI {period}",
                data=data_list,
                color=params.get("color", "#ff00aa"),
                width=2
            )
        ]
