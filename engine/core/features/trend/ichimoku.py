from typing import Dict, Any, List
import pandas as pd
import numpy as np
from ..base import Feature, FeatureResult, OutputSchema, OutputType, Pane, register_feature


@register_feature("Ichimoku")
class IchimokuCloud(Feature):
    @property
    def name(self) -> str:
        return "Ichimoku Cloud"

    @property
    def description(self) -> str:
        return "Comprehensive indicator defining support, resistance, and trend direction."

    @property
    def category(self) -> str:
        return "Trend"

    @property
    def output_schema(self) -> List[OutputSchema]:
        return [
            OutputSchema(name="tenkan", output_type=OutputType.LINE, pane=Pane.OVERLAY),
            OutputSchema(name="kijun", output_type=OutputType.LINE, pane=Pane.OVERLAY),
            OutputSchema(name="chikou", output_type=OutputType.LINE, pane=Pane.OVERLAY),
            OutputSchema(name="senkou_a", output_type=OutputType.LINE, pane=Pane.OVERLAY),
            OutputSchema(name="senkou_b", output_type=OutputType.LINE, pane=Pane.OVERLAY),
            OutputSchema(name="cloud", output_type=OutputType.BAND, pane=Pane.OVERLAY,
                         band_pair=("senkou_a", "senkou_b")),
        ]

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "conversion_period": 9,
            "base_period": 26,
            "lagging_span2_period": 52,
            "displacement": 26,
            "normalize": "none",
        }

    @property
    def parameter_options(self) -> Dict[str, Dict[str, Any]]:
        return {
            "conversion_period": {"min": 5, "max": 30, "step": 1},
            "base_period": {"min": 10, "max": 60, "step": 1},
            "lagging_span2_period": {"min": 20, "max": 120, "step": 1},
            "displacement": {"min": 10, "max": 60, "step": 1},
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any], cache: Any = None) -> FeatureResult:
        conv = int(params.get("conversion_period", 9))
        base = int(params.get("base_period", 26))
        span2 = int(params.get("lagging_span2_period", 52))
        disp = int(params.get("displacement", 26))
        norm_method = params.get("normalize", "none")

        high = df['High'] if 'High' in df.columns else df['high']
        low = df['Low'] if 'Low' in df.columns else df['low']
        close = df['Close'] if 'Close' in df.columns else df['close']

        # Tenkan-sen (Conversion Line)
        tenkan_sen = (high.rolling(window=conv).max() + low.rolling(window=conv).min()) / 2

        # Kijun-sen (Base Line)
        kijun_sen = (high.rolling(window=base).max() + low.rolling(window=base).min()) / 2

        # Senkou Span A (Leading Span A) shifted forward
        senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(disp)

        # Senkou Span B (Leading Span B) shifted forward
        senkou_b = ((high.rolling(window=span2).max() + low.rolling(window=span2).min()) / 2).shift(disp)

        # Chikou Span (Lagging Span) shifted backwards
        chikou_span = close.shift(-disp)

        col_tenkan = self.generate_column_name("Ichimoku", params, "tenkan")
        col_kijun = self.generate_column_name("Ichimoku", params, "kijun")
        col_chikou = self.generate_column_name("Ichimoku", params, "chikou")
        col_senkou_a = self.generate_column_name("Ichimoku", params, "senkou_a")
        col_senkou_b = self.generate_column_name("Ichimoku", params, "senkou_b")

        data = {
            col_tenkan: self.normalize(df, tenkan_sen, norm_method),
            col_kijun: self.normalize(df, kijun_sen, norm_method),
            col_chikou: self.normalize(df, chikou_span, norm_method),
            col_senkou_a: self.normalize(df, senkou_a, norm_method),
            col_senkou_b: self.normalize(df, senkou_b, norm_method),
        }

        return FeatureResult(data=data)
