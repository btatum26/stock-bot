from typing import Dict, Any, List
import pandas as pd
import numpy as np
from scipy.stats import linregress
from ..base import Feature, FeatureResult, OutputSchema, OutputType, Pane, register_feature


@register_feature("LinReg")
class LinearRegressionChannel(Feature):
    @property
    def name(self) -> str:
        return "Linear Regression Channel"

    @property
    def description(self) -> str:
        return "Linear Regression Channel with Standard Deviation bands."

    @property
    def category(self) -> str:
        return "Trend"

    @property
    def output_schema(self) -> List[OutputSchema]:
        return [
            OutputSchema(name="basis", output_type=OutputType.LINE, pane=Pane.OVERLAY),
            OutputSchema(name="upper", output_type=OutputType.LINE, pane=Pane.OVERLAY),
            OutputSchema(name="lower", output_type=OutputType.LINE, pane=Pane.OVERLAY),
            OutputSchema(name="channel", output_type=OutputType.BAND, pane=Pane.OVERLAY,
                         band_pair=("upper", "lower")),
        ]

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "lookback": 100,
            "std_dev": 2.0,
            "normalize": "none",
        }

    @property
    def parameter_options(self) -> Dict[str, Dict[str, Any]]:
        return {
            "lookback": {"min": 20, "max": 500, "step": 10},
            "std_dev": {"min": 0.5, "max": 4.0, "step": 0.5},
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any], cache: Any = None) -> FeatureResult:
        lookback = int(params.get("lookback", 100))
        std_dev_mult = float(params.get("std_dev", 2.0))
        norm_method = params.get("normalize", "none")

        close = df['Close'] if 'Close' in df.columns else df['close']

        if len(df) < 2:
            empty = pd.Series(np.nan, index=df.index)
            col_basis = self.generate_column_name("LinReg", params, "basis")
            col_upper = self.generate_column_name("LinReg", params, "upper")
            col_lower = self.generate_column_name("LinReg", params, "lower")
            return FeatureResult(data={col_basis: empty, col_upper: empty, col_lower: empty})

        if lookback > 0 and len(df) > lookback:
            start_idx = len(df) - lookback
            subset_close = close.iloc[start_idx:]
        else:
            start_idx = 0
            subset_close = close

        x = np.arange(len(subset_close))
        y = subset_close.values

        slope, intercept, r_value, p_value, std_err = linregress(x, y)

        reg_line = slope * x + intercept
        residuals = y - reg_line
        std = np.std(residuals)

        upper = reg_line + (std * std_dev_mult)
        lower = reg_line - (std * std_dev_mult)

        # Build full-length series padded with NaN
        basis_series = pd.Series(np.nan, index=df.index)
        upper_series = pd.Series(np.nan, index=df.index)
        lower_series = pd.Series(np.nan, index=df.index)

        basis_series.iloc[start_idx:] = reg_line
        upper_series.iloc[start_idx:] = upper
        lower_series.iloc[start_idx:] = lower

        col_basis = self.generate_column_name("LinReg", params, "basis")
        col_upper = self.generate_column_name("LinReg", params, "upper")
        col_lower = self.generate_column_name("LinReg", params, "lower")

        return FeatureResult(data={
            col_basis: self.normalize(df, basis_series, norm_method),
            col_upper: self.normalize(df, upper_series, norm_method),
            col_lower: self.normalize(df, lower_series, norm_method),
        })
