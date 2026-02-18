from typing import Dict, Any, List
import pandas as pd
from .base import Feature, FeatureOutput, LineOutput

class IchimokuCloud(Feature):
    @property
    def name(self) -> str:
        return "Ichimoku Cloud"

    @property
    def description(self) -> str:
        return "Comprehensive indicator defining support, resistance, and trend direction."

    @property
    def category(self) -> str:
        return "Trend Indicators"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "conversion_period": 9,
            "base_period": 26,
            "lagging_span2_period": 52,
            "displacement": 26
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> List[FeatureOutput]:
        conv = int(params.get("conversion_period", 9))
        base = int(params.get("base_period", 26))
        span2 = int(params.get("lagging_span2_period", 52))
        disp = int(params.get("displacement", 26))
        
        high = df['High']
        low = df['Low']
        
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
        tenkan_sen = (high.rolling(window=conv).max() + low.rolling(window=conv).min()) / 2

        # Kijun-sen (Base Line): (26-period high + 26-period low)/2
        kijun_sen = (high.rolling(window=base).max() + low.rolling(window=base).min()) / 2

        # Senkou Span A (Leading Span A): (Conversion + Base)/2
        # Shifted forward by displacement
        senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(disp)

        # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
        # Shifted forward by displacement
        senkou_b = ((high.rolling(window=span2).max() + low.rolling(window=span2).min()) / 2).shift(disp)

        # Chikou Span (Lagging Span): Close shifted backwards
        chikou_span = df['Close'].shift(-disp)

        def clean(series):
            return series.where(pd.notnull(series), None).tolist()

        return [
            LineOutput(name="Tenkan-sen (Conv)", data=clean(tenkan_sen), color="#0496ff", width=1),
            LineOutput(name="Kijun-sen (Base)", data=clean(kijun_sen), color="#991515", width=1),
            LineOutput(name="Senkou Span A", data=clean(senkou_a), color="#00ff00", width=1),
            LineOutput(name="Senkou Span B", data=clean(senkou_b), color="#ff0000", width=1),
            LineOutput(name="Chikou Span (Lag)", data=clean(chikou_span), color="#ffffff", width=1)
        ]
