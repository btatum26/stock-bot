from typing import Dict, Any, List
import pandas as pd
from .base import Feature, FeatureOutput, LineOutput

class KeltnerChannels(Feature):
    @property
    def name(self) -> str:
        return "Keltner Channels"

    @property
    def description(self) -> str:
        return "Volatility bands based on ATR placed around an EMA."

    @property
    def category(self) -> str:
        return "Volatility"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "period": 20,
            "atr_period": 10,
            "multiplier": 2.0,
            "color": "#00ffaa"
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> List[FeatureOutput]:
        period = int(params.get("period", 20))
        atr_period = int(params.get("atr_period", 10))
        mult = float(params.get("multiplier", 2.0))
        color = params.get("color", "#00ffaa")
        
        # EMA Basis
        ema = df['Close'].ewm(span=period, adjust=False).mean()
        
        # ATR Calculation
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=atr_period).mean()
        
        upper = ema + (atr * mult)
        lower = ema - (atr * mult)
        
        def clean(s): return s.where(pd.notnull(s), None).tolist()
        
        return [
            LineOutput(name="Upper Keltner", data=clean(upper), color=color, width=1),
            LineOutput(name="Lower Keltner", data=clean(lower), color=color, width=1),
            LineOutput(name="Keltner Basis", data=clean(ema), color=color, width=1)
        ]
