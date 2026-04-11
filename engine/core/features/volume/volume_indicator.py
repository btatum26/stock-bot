from typing import Dict, Any, List
import pandas as pd
import numpy as np
from ..base import Feature, FeatureResult, OutputSchema, OutputType, Pane, register_feature


@register_feature("Volume")
class VolumeIndicator(Feature):
    @property
    def name(self) -> str:
        return "Volume"

    @property
    def description(self) -> str:
        return "Trading volume with optional moving average."

    @property
    def category(self) -> str:
        return "Volume"

    @property
    def output_schema(self) -> List[OutputSchema]:
        return [
            OutputSchema(name=None, output_type=OutputType.LINE, pane=Pane.NEW),
        ]

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "normalize": ["none", "z_score", "pct_distance", "price_ratio"],
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any], cache: Any = None) -> FeatureResult:
        norm_method = params.get("normalize", "none")

        volume = df['Volume'] if 'Volume' in df.columns else df['volume']

        col_name = self.generate_column_name("Volume", params)
        final_data = self.normalize(df, volume, norm_method)

        return FeatureResult(data={col_name: final_data})
