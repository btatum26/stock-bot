from typing import Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
from .base import Feature, FeatureOutput, HeatmapOutput

class KernelDensityEst(Feature):
    @property
    def name(self) -> str:
        return "Kernel Density Heatmap"

    @property
    def description(self) -> str:
        return "Visualizes price density as a continuous background gradient."

    @property
    def category(self) -> str:
        return "Volume & Profile"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "bandwidth": 1.0, 
            "source": ["Close", "High", "Low", "Open"]
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> List[FeatureOutput]:
        bandwidth = float(params.get("bandwidth", 1.0))
        source_col = params.get("source", "Close")
        
        prices = df[source_col].dropna().values.reshape(-1, 1)
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
        kde.fit(prices)
        
        # 1000 points resolution for the heatmap
        min_price = prices.min()
        max_price = prices.max()
        price_grid = np.linspace(min_price, max_price, 1000).reshape(-1, 1)
        
        log_density = kde.score_samples(price_grid)
        density = np.exp(log_density)
        
        # Normalize density 0-1 for visualization
        density = (density - density.min()) / (density.max() - density.min())
        
        return [
            HeatmapOutput(
                name="Price Density",
                price_grid=price_grid.flatten().tolist(),
                density=density.flatten().tolist(),
                color_map="blue" # Simple blue gradient
            )
        ]
