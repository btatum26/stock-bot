from typing import Dict, Any, List
import pandas as pd
import numpy as np
from ..base import Feature, FeatureResult, OutputSchema, OutputType, Pane, register_feature


@register_feature("VolumeProfile")
class VolumeProfile(Feature):
    @property
    def name(self) -> str:
        return "Volume Profile (VPVR)"

    @property
    def description(self) -> str:
        return "Horizontal volume histogram showing high-volume price nodes."

    @property
    def category(self) -> str:
        return "Volume"

    @property
    def output_schema(self) -> List[OutputSchema]:
        return [
            OutputSchema(name="profile", output_type=OutputType.HEATMAP, pane=Pane.OVERLAY),
        ]

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "bins": 100,
            "lookback": 0,
        }

    @property
    def parameter_options(self) -> Dict[str, Dict[str, Any]]:
        return {
            "bins": {"min": 20, "max": 500, "step": 10},
            "lookback": {"min": 0, "max": 1000, "step": 50},
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any], cache: Any = None) -> FeatureResult:
        bins = int(params.get("bins", 100))
        lookback = int(params.get("lookback", 0))

        if lookback > 0 and len(df) > lookback:
            subset = df.iloc[-lookback:]
        else:
            subset = df

        low = subset['Low'] if 'Low' in subset.columns else subset['low']
        high = subset['High'] if 'High' in subset.columns else subset['high']
        close = subset['Close'] if 'Close' in subset.columns else subset['close']
        volume = subset['Volume'] if 'Volume' in subset.columns else subset['volume']

        min_price = float(low.min())
        max_price = float(high.max())

        hist, bin_edges = np.histogram(
            close, bins=bins, range=(min_price, max_price), weights=volume
        )

        # Normalize to 0-1
        if hist.max() > 0:
            hist_norm = hist / hist.max()
        else:
            hist_norm = hist.astype(float)

        # Price grid = center of each bin
        price_grid = (bin_edges[:-1] + bin_edges[1:]) / 2

        return FeatureResult(heatmaps={
            "profile": {
                "price_grid": price_grid,
                "time_index": subset.index,
                "intensity": hist_norm,
            }
        })
