from typing import Dict, Any, List
import numpy as np
import pandas as pd
from ..base import Feature, FeatureResult, OutputSchema, OutputType, Pane, register_feature


@register_feature("YearlyCycle")
class YearlyCycle(Feature):
    @property
    def name(self) -> str:
        return "Yearly Cycle"

    @property
    def description(self) -> str:
        return (
            "Cyclical sine/cosine encoding of day-of-year (period 365.25). "
            "Captures seasonal/annual effects in a continuous, wrap-around form suitable for ML."
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

        # dayofyear is 1..366; subtract 1 so the cycle starts at angle 0 on Jan 1.
        day = index.dayofyear.to_numpy(dtype=np.float64) - 1.0
        angle = 2.0 * np.pi * day / 365.25

        sin_series = pd.Series(np.sin(angle), index=df.index)
        cos_series = pd.Series(np.cos(angle), index=df.index)

        col_sin = self.generate_column_name("YearlyCycle", params, "sin")
        col_cos = self.generate_column_name("YearlyCycle", params, "cos")

        return FeatureResult(data={col_sin: sin_series, col_cos: cos_series})
