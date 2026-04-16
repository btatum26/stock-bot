from typing import Dict, Any, List
import pandas as pd
import numpy as np
from ..base import Feature, FeatureResult, OutputSchema, OutputType, Pane, register_feature


@register_feature("Supertrend")
class Supertrend(Feature):
    @property
    def name(self) -> str:
        return "Supertrend"

    @property
    def description(self) -> str:
        return "Trend-following indicator using ATR to set stop-loss levels."

    @property
    def category(self) -> str:
        return "Trend"

    @property
    def output_schema(self) -> List[OutputSchema]:
        return [
            OutputSchema(name="line", output_type=OutputType.LINE, pane=Pane.OVERLAY),
            OutputSchema(name="direction", output_type=OutputType.LINE, pane=Pane.NEW, y_range=(-1, 1)),
        ]

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "period": 10,
            "multiplier": 3.0,
            "normalize": ["none", "z_score", "pct_distance", "price_ratio"],
        }

    @property
    def parameter_options(self) -> Dict[str, Dict[str, Any]]:
        return {
            "period": {"min": 5, "max": 50, "step": 1},
            "multiplier": {"min": 1.0, "max": 6.0, "step": 0.5},
        }

    def non_stationary_outputs(self, params: Dict[str, Any]) -> List[str]:
        # The supertrend line is a price-coupled stop level; direction is a
        # bounded +/-1 flag. Only FFD the line.
        if params.get("normalize", "none") != "none":
            return []
        return [self.generate_column_name("Supertrend", params, "line")]

    def compute(self, df: pd.DataFrame, params: Dict[str, Any], cache: Any = None) -> FeatureResult:
        period = int(params.get("period", 10))
        multiplier = float(params.get("multiplier", 3.0))
        norm_method = params.get("normalize", "none")

        high = df['High'] if 'High' in df.columns else df['high']
        low = df['Low'] if 'Low' in df.columns else df['low']
        close = df['Close'] if 'Close' in df.columns else df['close']

        # Calculate ATR
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()

        # Basic Bands
        hl2 = (high + low) / 2
        basic_upper = hl2 + (multiplier * atr)
        basic_lower = hl2 - (multiplier * atr)

        # Final Bands (iterative logic required)
        n = len(df)
        final_upper = np.zeros(n)
        final_lower = np.zeros(n)
        supertrend = np.full(n, np.nan)
        trend = np.ones(n, dtype=int)  # 1 = Up, -1 = Down

        close_vals = close.values
        bu_vals = basic_upper.values
        bl_vals = basic_lower.values

        for i in range(period, n):
            if bu_vals[i] < final_upper[i - 1] or close_vals[i - 1] > final_upper[i - 1]:
                final_upper[i] = bu_vals[i]
            else:
                final_upper[i] = final_upper[i - 1]

            if bl_vals[i] > final_lower[i - 1] or close_vals[i - 1] < final_lower[i - 1]:
                final_lower[i] = bl_vals[i]
            else:
                final_lower[i] = final_lower[i - 1]

            # Trend switch logic
            if trend[i - 1] == 1:
                trend[i] = -1 if close_vals[i] < final_lower[i - 1] else 1
            else:
                trend[i] = 1 if close_vals[i] > final_upper[i - 1] else -1

            supertrend[i] = final_lower[i] if trend[i] == 1 else final_upper[i]

        supertrend_series = pd.Series(supertrend, index=df.index)
        trend_series = pd.Series(trend, index=df.index, dtype=float)
        # Mark warm-up period as NaN
        trend_series.iloc[:period] = np.nan

        col_line = self.generate_column_name("Supertrend", params, "line")
        col_dir = self.generate_column_name("Supertrend", params, "direction")

        return FeatureResult(data={
            col_line: self.normalize(df, supertrend_series, norm_method),
            col_dir: trend_series,
        })
