from typing import Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
from ..base import Feature, FeatureResult, OutputSchema, OutputType, Pane, register_feature


@register_feature("KDE")
class KernelDensityEstimation(Feature):
    @property
    def name(self) -> str:
        return "Kernel Density Heatmap"

    @property
    def description(self) -> str:
        return "Price density estimation using Gaussian kernel density."

    @property
    def category(self) -> str:
        return "Price Levels"

    @property
    def output_schema(self) -> List[OutputSchema]:
        return [
            OutputSchema(name="density", output_type=OutputType.HEATMAP, pane=Pane.OVERLAY),
        ]

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "bandwidth": 1.0,
            "source": "Close",
            "resolution": 1000,
        }

    @property
    def parameter_options(self) -> Dict[str, Dict[str, Any]]:
        return {
            "bandwidth": {"min": 0.1, "max": 10.0, "step": 0.1},
            "source": {"options": ["Close", "High", "Low", "Open"]},
            "resolution": {"min": 100, "max": 2000, "step": 100},
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any], cache: Any = None) -> FeatureResult:
        bandwidth = float(params.get("bandwidth", 1.0))
        source_col = params.get("source", "Close")
        resolution = int(params.get("resolution", 1000))

        prices = df[source_col].dropna().values.reshape(-1, 1)

        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
        kde.fit(prices)

        min_price = float(prices.min())
        max_price = float(prices.max())
        price_grid = np.linspace(min_price, max_price, resolution).reshape(-1, 1)

        log_density = kde.score_samples(price_grid)
        density = np.exp(log_density)

        # Normalize density to 0-1
        d_min, d_max = density.min(), density.max()
        if d_max > d_min:
            density = (density - d_min) / (d_max - d_min)

        return FeatureResult(heatmaps={
            "density": {
                "price_grid": price_grid.flatten(),
                "time_index": df.index,
                "intensity": density.flatten(),
            }
        })
