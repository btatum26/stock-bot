from typing import Dict, Any, List
import pandas as pd
from ..base import Feature, FeatureResult, register_feature

@register_feature("ROC")
class ROC(Feature):
    @property
    def name(self) -> str:
        return "ROC"

    @property
    def description(self) -> str:
        return "Rate of Change (Percentage difference between current and n-period ago price)."

    @property
    def category(self) -> str:
        return "Oscillators (Momentum)"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "period": 12,
            "normalize": "none"
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any], cache: Any = None) -> FeatureResult:
        period = int(params.get("period", 12))
        norm_method = params.get("normalize", "none")
        
        close = df['Close'] if 'Close' in df.columns else df['close']
        
        # ROC Calculation
        # ((Current Close - Close n periods ago) / Close n periods ago) * 100
        roc = close.pct_change(periods=period) * 100
        
        col_name = self.generate_column_name("ROC", params)
        
        # Apply systematic normalization
        final_data = self.normalize(df, roc, norm_method)
        
        return FeatureResult(data={col_name: final_data})
