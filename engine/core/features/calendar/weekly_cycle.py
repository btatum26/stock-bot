from typing import Dict, Any, List
import numpy as np
import pandas as pd
from ..base import Feature, FeatureResult, OutputSchema, OutputType, Pane, register_feature


@register_feature("WeeklyCycle")
class WeeklyCycle(Feature):
    @property
    def name(self) -> str:
        return "Weekly Cycle"

    @property
    def description(self) -> str:
        return (
            "Cyclical sine/cosine encoding of day-of-week (period 7). "
            "Captures weekday effects in a continuous, wrap-around form suitable for ML."
        )

    @property
    def category(self) -> str:
        return "Calendar"

    @property
    def output_schema(self) -> List[OutputSchema]:
        return [
            OutputSchema(name="sin", output_type=OutputType.LINE, pane=Pane.NEW, y_range=(-1.0, 1.0)),
            OutputSchema(name="cos", output_type=OutputType.LINE, pane=Pane.NEW, y_range=(-1.0, 1.0)),
        ]

    def compute(self, df: pd.DataFrame, params: Dict[str, Any], cache: Any = None) -> FeatureResult:
        index = df.index
        if not isinstance(index, pd.DatetimeIndex):
            index = pd.to_datetime(index)

        angle = 2.0 * np.pi * index.dayofweek.to_numpy(dtype=np.float64) / 7.0

        sin_series = pd.Series(np.sin(angle), index=df.index)
        cos_series = pd.Series(np.cos(angle), index=df.index)

        col_sin = self.generate_column_name("WeeklyCycle", params, "sin")
        col_cos = self.generate_column_name("WeeklyCycle", params, "cos")

        return FeatureResult(data={col_sin: sin_series, col_cos: cos_series})
