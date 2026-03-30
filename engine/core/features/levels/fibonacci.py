from typing import Dict, Any, List
import pandas as pd
import numpy as np
from ..base import Feature, FeatureResult, OutputSchema, OutputType, Pane, register_feature


@register_feature("Fibonacci")
class FibonacciRetracement(Feature):
    @property
    def name(self) -> str:
        return "Fibonacci Retracement"

    @property
    def description(self) -> str:
        return "Horizontal levels based on the High-Low range of the lookback period."

    @property
    def category(self) -> str:
        return "Price Levels"

    @property
    def output_schema(self) -> List[OutputSchema]:
        return [
            OutputSchema(name="levels", output_type=OutputType.LEVEL, pane=Pane.OVERLAY),
        ]

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "lookback": 0,
        }

    @property
    def parameter_options(self) -> Dict[str, Dict[str, Any]]:
        return {
            "lookback": {"min": 0, "max": 1000, "step": 10},
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any], cache: Any = None) -> FeatureResult:
        lookback = int(params.get("lookback", 0))

        high = df['High'] if 'High' in df.columns else df['high']
        low = df['Low'] if 'Low' in df.columns else df['low']

        if lookback > 0 and len(df) > lookback:
            window_high = high.iloc[-lookback:]
            window_low = low.iloc[-lookback:]
        else:
            window_high = high
            window_low = low

        high_price = float(window_high.max())
        low_price = float(window_low.min())
        diff = high_price - low_price

        fib_ratios = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]

        levels = []
        for ratio in fib_ratios:
            price = high_price - (diff * ratio)
            levels.append({
                "value": round(price, 4),
                "label": f"Fib {ratio:.3f}",
                "strength": 1.0,
            })

        return FeatureResult(levels=levels)
