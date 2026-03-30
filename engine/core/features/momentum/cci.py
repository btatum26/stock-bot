from typing import Dict, Any, List
import pandas as pd
import numpy as np
from ..base import Feature, FeatureResult, OutputSchema, OutputType, Pane, register_feature


@register_feature("CCI")
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
    def output_schema(self) -> List[OutputSchema]:
        return [
            OutputSchema(name=None, output_type=OutputType.LINE, pane=Pane.NEW),
            OutputSchema(name="overbought", output_type=OutputType.LEVEL, pane=Pane.NEW),
            OutputSchema(name="oversold", output_type=OutputType.LEVEL, pane=Pane.NEW),
        ]

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "period": 20,
            "overbought": 100,
            "oversold": -100,
            "normalize": "none",
        }

    @property
    def parameter_options(self) -> Dict[str, Dict[str, Any]]:
        return {
            "period": {"min": 5, "max": 100, "step": 1},
            "overbought": {"min": 50, "max": 300, "step": 10},
            "oversold": {"min": -300, "max": -50, "step": 10},
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any], cache: Any = None) -> FeatureResult:
        period = int(params.get("period", 20))
        ob = float(params.get("overbought", 100))
        os_ = float(params.get("oversold", -100))
        norm_method = params.get("normalize", "none")

        high = df['High'] if 'High' in df.columns else df['high']
        low = df['Low'] if 'Low' in df.columns else df['low']
        close = df['Close'] if 'Close' in df.columns else df['close']

        tp = (high + low + close) / 3
        sma = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))

        cci = (tp - sma) / (0.015 * mad)

        col_name = self.generate_column_name("CCI", params)
        final_data = self.normalize(df, cci, norm_method)

        return FeatureResult(
            data={col_name: final_data},
            levels=[
                {"value": ob, "label": "Overbought"},
                {"value": os_, "label": "Oversold"},
            ],
        )
