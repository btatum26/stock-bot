from typing import Dict, Any, List
import pandas as pd
from ..base import Feature, FeatureResult, OutputSchema, OutputType, Pane, register_feature

@register_feature("VolumeZScore")
class VolumeZScore(Feature):
    @property
    def name(self) -> str:
        return "Volume Z-Score"

    @property
    def description(self) -> str:
        return "Identifies abnormal volume spikes using Z-Score normalization."

    @property
    def category(self) -> str:
        return "Volume Indicators"

    @property
    def output_schema(self) -> List[OutputSchema]:
        return [
            OutputSchema(name=None,          output_type=OutputType.LINE,  pane=Pane.NEW),
            OutputSchema(name="high_volume", output_type=OutputType.LEVEL, pane=Pane.NEW),
            OutputSchema(name="low_volume",  output_type=OutputType.LEVEL, pane=Pane.NEW),
        ]

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "period": 20
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any], cache: Any = None) -> FeatureResult:
        period = int(params.get("period", 20))
        
        volume = df['Volume'] if 'Volume' in df.columns else df['volume']
        
        # Calculate Z-Score manually or via base class normalization
        # Here we do it manually to ensure we are using Volume
        z_score = self.normalize(df, volume, "z_score")
        
        col_name = self.generate_column_name("VolumeZScore", params)
        
        return FeatureResult(data={col_name: z_score})
