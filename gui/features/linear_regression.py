from typing import Dict, Any, List
import pandas as pd
import numpy as np
from scipy.stats import linregress
from .base import Feature, FeatureOutput, LineOutput

class LinearRegressionChannel(Feature):
    @property
    def name(self) -> str:
        return "LinReg Channel"

    @property
    def description(self) -> str:
        return "Linear Regression Channel with Standard Deviation bands."

    @property
    def category(self) -> str:
        return "Trend Indicators"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "lookback": 100,
            "std_dev": 2.0,
            "color": "#00ffff"
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> List[FeatureOutput]:
        lookback = int(params.get("lookback", 100))
        std_dev = float(params.get("std_dev", 2.0))
        color = params.get("color", "#00ffff")
        
        if len(df) < 2:
            return []
            
        # Determine subset to analyze
        if lookback > 0 and len(df) > lookback:
            start_idx = len(df) - lookback
            subset = df.iloc[start_idx:]
        else:
            start_idx = 0
            subset = df
            
        x = np.arange(len(subset))
        y = subset['Close'].values
        
        # Calculate Linear Regression
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        
        # Calculate Regression Line
        reg_line = slope * x + intercept
        
        # Calculate Standard Deviation from the line
        residuals = y - reg_line
        std = np.std(residuals)
        
        upper = reg_line + (std * std_dev)
        lower = reg_line - (std * std_dev)
        
        # Pad the beginning with None so it aligns with the chart
        padding = [None] * start_idx
        
        return [
            LineOutput(name="LinReg Basis", data=padding + reg_line.tolist(), color=color, width=2),
            LineOutput(name="Upper Channel", data=padding + upper.tolist(), color=color, width=1),
            LineOutput(name="Lower Channel", data=padding + lower.tolist(), color=color, width=1)
        ]
