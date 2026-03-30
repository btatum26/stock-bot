from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from ..base import Feature, FeatureResult, register_feature
from .rsi import RSI

@register_feature("RSI_Divergence_Features")
class RSIDivergenceFeatures(Feature):
    """
    Computes all necessary data for RSI Fractal Divergence strategy.
    Implements T-Zero rule by shifting fractal confirmation.
    """
    @property
    def name(self) -> str:
        return "RSI Divergence Features"

    @property
    def description(self) -> str:
        return "T-Zero compliant RSI fractal divergence features."

    @property
    def category(self) -> str:
        return "Oscillators (Momentum)"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "rsi_period": 14,
            "fractal_n": 5, # Distance on each side of the peak/valley
        }

    @property
    def outputs(self) -> List[str]:
        return [
            "RSI_Base",
            "Is_Fractal_High_Confirmed",
            "Is_Fractal_Low_Confirmed",
            "Current_Confirmed_High_Price",
            "Current_Confirmed_High_RSI",
            "Current_Confirmed_Low_Price",
            "Current_Confirmed_Low_RSI",
            "Previous_High_Price",
            "Previous_High_RSI",
            "Previous_Low_Price",
            "Previous_Low_RSI",
            "Bars_Since_Previous_High",
            "Bars_Since_Previous_Low"
        ]

    def compute(self, df: pd.DataFrame, params: Dict[str, Any], cache: Any = None) -> FeatureResult:
        rsi_period = int(params.get("rsi_period", 14))
        n = int(params.get("fractal_n", 5))
        
        # 1. Base RSI
        # Try to get from cache first
        rsi_feature = RSI()
        rsi_params = {"period": rsi_period}
        rsi_col_base = rsi_feature.generate_column_name("RSI", rsi_params)
        
        if cache and rsi_col_base in cache._memory:
            rsi_base = cache._memory[rsi_col_base]
        else:
            rsi_result = rsi_feature.compute(df, rsi_params, cache)
            rsi_base = rsi_result.data[rsi_col_base]
            if cache:
                cache.set_series(rsi_col_base, rsi_base)

        high = df['High'] if 'High' in df.columns else df['high']
        low = df['Low'] if 'Low' in df.columns else df['low']
        close = df['Close'] if 'Close' in df.columns else df['close']

        # pd.Series.rolling(window=2N+1, center=True) identifies the peak at T-N
        # We confirm it at T (current)
        window = 2 * n + 1
        
        # Fractal High
        is_peak = (high == high.rolling(window=window, center=True).max())
        is_fractal_high_confirmed = is_peak.shift(n).fillna(False)
        
        # Fractal Low
        is_valley = (low == low.rolling(window=window, center=True).min())
        is_fractal_low_confirmed = is_valley.shift(n).fillna(False)

        # Note: Values are from T-n (the actual fractal peak/valley) but stamped at T
        current_confirmed_high_price = high.shift(n).where(is_fractal_high_confirmed)
        current_confirmed_high_rsi = rsi_base.shift(n).where(is_fractal_high_confirmed)
        
        current_confirmed_low_price = low.shift(n).where(is_fractal_low_confirmed)
        current_confirmed_low_rsi = rsi_base.shift(n).where(is_fractal_low_confirmed)

        # We want to compare the NEWLY confirmed fractal with the PREVIOUSLY confirmed one.
        
        # Highs
        # Extract only confirmed values to a new series, shift it, then re-align and ffill
        high_indices = is_fractal_high_confirmed[is_fractal_high_confirmed].index
        
        # Previous High Price
        prev_high_price_vals = current_confirmed_high_price.loc[high_indices].shift(1)
        previous_high_price = pd.Series(index=df.index, dtype=float)
        previous_high_price.loc[high_indices] = prev_high_price_vals
        previous_high_price = previous_high_price.ffill()
        
        # Previous High RSI
        prev_high_rsi_vals = current_confirmed_high_rsi.loc[high_indices].shift(1)
        previous_high_rsi = pd.Series(index=df.index, dtype=float)
        previous_high_rsi.loc[high_indices] = prev_high_rsi_vals
        previous_high_rsi = previous_high_rsi.ffill()
        
        # Lows
        low_indices = is_fractal_low_confirmed[is_fractal_low_confirmed].index
        
        # Previous Low Price
        prev_low_price_vals = current_confirmed_low_price.loc[low_indices].shift(1)
        previous_low_price = pd.Series(index=df.index, dtype=float)
        previous_low_price.loc[low_indices] = prev_low_price_vals
        previous_low_price = previous_low_price.ffill()
        
        # Previous Low RSI
        prev_low_rsi_vals = current_confirmed_low_rsi.loc[low_indices].shift(1)
        previous_low_rsi = pd.Series(index=df.index, dtype=float)
        previous_low_rsi.loc[low_indices] = prev_low_rsi_vals
        previous_low_rsi = previous_low_rsi.ffill()

        # Highs counter
        high_conf_indices = np.where(is_fractal_high_confirmed)[0]
        full_indices = np.arange(len(df))
        last_high_idx = pd.Series(np.nan, index=df.index)
        last_high_idx.iloc[high_conf_indices] = high_conf_indices
        last_high_idx = last_high_idx.ffill().shift(1)
        bars_since_previous_high = full_indices - last_high_idx

        # Lows counter
        low_conf_indices = np.where(is_fractal_low_confirmed)[0]
        last_low_idx = pd.Series(np.nan, index=df.index)
        last_low_idx.iloc[low_conf_indices] = last_low_idx
        last_low_idx = last_low_idx.ffill().shift(1)
        bars_since_previous_low = full_indices - last_low_idx

        # Prepare Data Dict with generated column names
        data = {
            self.generate_column_name("RSI_Divergence_Features", params, "RSI_Base"): rsi_base,
            self.generate_column_name("RSI_Divergence_Features", params, "Is_Fractal_High_Confirmed"): is_fractal_high_confirmed,
            self.generate_column_name("RSI_Divergence_Features", params, "Is_Fractal_Low_Confirmed"): is_fractal_low_confirmed,
            self.generate_column_name("RSI_Divergence_Features", params, "Current_Confirmed_High_Price"): current_confirmed_high_price,
            self.generate_column_name("RSI_Divergence_Features", params, "Current_Confirmed_High_RSI"): current_confirmed_high_rsi,
            self.generate_column_name("RSI_Divergence_Features", params, "Current_Confirmed_Low_Price"): current_confirmed_low_price,
            self.generate_column_name("RSI_Divergence_Features", params, "Current_Confirmed_Low_RSI"): current_confirmed_low_rsi,
            self.generate_column_name("RSI_Divergence_Features", params, "Previous_High_Price"): previous_high_price,
            self.generate_column_name("RSI_Divergence_Features", params, "Previous_High_RSI"): previous_high_rsi,
            self.generate_column_name("RSI_Divergence_Features", params, "Previous_Low_Price"): previous_low_price,
            self.generate_column_name("RSI_Divergence_Features", params, "Previous_Low_RSI"): previous_low_rsi,
            self.generate_column_name("RSI_Divergence_Features", params, "Bars_Since_Previous_High"): bars_since_previous_high,
            self.generate_column_name("RSI_Divergence_Features", params, "Bars_Since_Previous_Low"): bars_since_previous_low
        }

        return FeatureResult(data=data)
