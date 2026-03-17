# --- AVAILABLE FEATURES ---
# This strategy has access to the following features in `feature_data`:
# - Volume (Always available)
# - Support & Resistance
# - RSI
# - ATR
# - Moving Average
# --------------------------

import pprint

import pandas as pd
import numpy as np
from src.signals.base import SignalModel
from typing import Dict, Any

class StrategySignal(SignalModel):
    def __init__(self):
        super().__init__()

    @property
    def signal_parameters(self) -> Dict[str, Any]:
        """
        Strategy-specific rules/parameters.
        """
        return {
            "oversold_threshold": 30.0,
            "lookback_window": 60,
            "pivot_window": 3,
            "min_pivot_change": 5.0,
            "min_level_dist": 0.01
        }

    @property
    def parameters(self) -> Dict[str, Any]:
        """Alias for backward compatibility."""
        return self.signal_parameters

    # --- INDIVIDUAL NORMALIZATION & HELPER FUNCTIONS ---

    def normalize_to_bar(self, df: pd.DataFrame, feature_data: dict, ref_idx: int) -> dict:
        """Normalizes price and indicators relative to the value at ref_idx."""
        normalized = {}
        ref_price = df['Close'].iloc[ref_idx]
        if ref_price == 0: ref_price = 1e-9
        
        normalized['Close'] = (df['Close'] - ref_price) / ref_price
        normalized['High'] = (df['High'] - ref_price) / ref_price
        normalized['Low'] = (df['Low'] - ref_price) / ref_price
        normalized['Open'] = (df['Open'] - ref_price) / ref_price

        for key, series in feature_data.items():
            if any(x in key for x in ['Dist', 'RSI', 'Ratio', 'Percent']):
                continue
            ref_val = series.iloc[ref_idx]
            if pd.isna(ref_val) or ref_val == 0:
                s_std = series.std()
                if pd.isna(s_std) or s_std == 0: s_std = 1.0
                normalized[key] = (series - series.mean()) / s_std
            else:
                normalized[key] = (series - ref_val) / ref_val
        return normalized

    def extract_sr_features(self, price: float, feature_data: dict, idx: int) -> dict:
        """Extracts Support/Resistance features."""
        sr_feat = {}
        found_dist = False
        for key, series in feature_data.items():
            if 'Dist_to_Support' in key:
                sr_feat['Dist_to_Support'] = series.iloc[idx]
                found_dist = True
            elif 'Dist_to_Resistance' in key:
                sr_feat['Dist_to_Resistance'] = series.iloc[idx]
                found_dist = True
        
        if not found_dist:
            levels = []
            for key, series in feature_data.items():
                if any(kw in key for kw in ['SR', 'Support', 'Resistance', 'Level']) and 'Dist' not in key:
                    val = series.iloc[idx]
                    if not pd.isna(val) and val != 0:
                        levels.append(val)
            sr_feat['Dist_Nearest_Level'] = min(abs(price - l) / price for l in levels) if levels else 0.0
        
        return sr_feat

    def normalize_mas(self, price: float, feature_data: dict, idx: int) -> dict:
        """Calculates percentage distance from price to Moving Averages."""
        ma_features = {}
        for key, series in feature_data.items():
            if 'MA' in key and 'Dist' not in key:
                val = series.iloc[idx]
                if not pd.isna(val) and val != 0:
                    ma_features[f"Dist_{key}"] = (price - val) / price
        return ma_features

    def volatility_scale_zscore(self, series: pd.Series, n: int = 20) -> pd.Series:
        rolling_mean = series.rolling(window=n).mean()
        rolling_std = series.rolling(window=n).std()
        return (series - rolling_mean) / rolling_std.replace(0, 1e-9)

    def volatility_scale_atr(self, price_series: pd.Series, atr_series: pd.Series) -> pd.Series:
        return (price_series.diff() / atr_series.replace(0, 1e-9)).fillna(0)

    def normalize_atr_to_price(self, atr_series: pd.Series, price_series: pd.Series) -> pd.Series:
        return (atr_series / price_series.replace(0, 1e-9)).fillna(0)

    # --- DIVERGENCE HELPERS ---

    def is_pivot_low(self, series: pd.Series, idx: int, window: int, min_change: float = 0.0) -> bool:
        """Checks if the bar at 'idx' is a local minimum within 'window' bars on each side."""
        if idx < window or idx >= len(series) - window:
            return False
        val = series.iloc[idx]
        for j in range(1, window + 1):
            if series.iloc[idx - j] <= val or series.iloc[idx + j] < val:
                return False
        
        if min_change > 0:
            # Check if there's a significant drop into the pivot and rise out of it
            left_max = series.iloc[idx - window : idx].max()
            right_max = series.iloc[idx + 1 : idx + window + 1].max()
            if (left_max - val) < min_change or (right_max - val) < min_change:
                return False
                
        return True

    def check_bullish_divergence(self, df: pd.DataFrame, rsi: pd.Series, pivot_idx: int, rsi_pivots: list, settings: dict) -> bool:
        """
        Core logic to detect Bullish RSI Divergence at a confirmed pivot point.
        Price: Lower Low | RSI: Higher Low (from Oversold)
        """
        oversold = settings.get("oversold_threshold", 30.0)
        lookback = settings.get("lookback_window", 60)
        
        current_rsi_low = rsi.iloc[pivot_idx]
        current_price_low = df['Low'].iloc[pivot_idx]

        if current_rsi_low < oversold:
            # Look back for the previous RSI pivot low within the lookback window
            prev_rsi_pivots = [p for p in rsi_pivots if p[0] >= pivot_idx - lookback]
            
            if prev_rsi_pivots:
                prev_idx, prev_rsi_low = prev_rsi_pivots[-1]
                prev_price_low = df['Low'].iloc[prev_idx]

                # --- BULLISH DIVERGENCE CONDITION ---
                price_ll = current_price_low < prev_price_low
                rsi_hl = current_rsi_low > prev_rsi_low

                return price_ll and rsi_hl
        
        return False

    def generate_signals(self, df: pd.DataFrame, feature_data: dict) -> pd.Series:
        """
        Entry: Standard Bullish RSI Divergence + ML Model Support.
        If an active ML model is set, it will use that for prediction.
        Otherwise, it falls back to the rule-based RSI divergence.
        """
        signals = pd.Series(0, index=df.index)
        
        # --- ML MODEL PREDICTION ---
        if hasattr(self, 'active_model_id') and self.active_model_id and self.active_model_id in self.model_instances:
            model = self.model_instances[self.active_model_id]['weights']
            
            # Prepare Features for ML
            X = pd.DataFrame(index=df.index)
            for k, v in feature_data.items(): X[k] = v
            X_clean = X.dropna()
            
            if not X_clean.empty:
                preds = model.predict(X_clean)
                signals.loc[X_clean.index] = preds
            return signals

        # --- RULE-BASED RSI DIVERGENCE ---
        rsi = next((v for k, v in feature_data.items() if 'RSI' in k), None)
        if rsi is None: return signals

        settings = self.parameters
        p_window = settings.get("pivot_window", 3)
        min_change = settings.get("min_pivot_change", 5.0)

        print("DEBUG: settings for RSI Divergence:")
        pprint.pprint(settings)
        # Track pivots discovered during the scan
        rsi_pivots = [] # List of (index, rsi_value)

        for i in range(p_window, len(df)):
            pivot_idx = i - p_window
            if self.is_pivot_low(rsi, pivot_idx, p_window, min_change):
                if self.check_bullish_divergence(df, rsi, pivot_idx, rsi_pivots, settings):
                    signals.iloc[i] = 1
                rsi_pivots.append((pivot_idx, rsi.iloc[pivot_idx]))

        return signals

    def calculate_exit_scores(self, df: pd.DataFrame, signals: pd.Series, alpha: float = 1.0, gamma: float = 0.15) -> pd.Series:
        """
        Calculates SellScore for each bar following a buy signal.
        This follows the formula: SellScore_k = (Gain_k / (k - t0)^gamma) - (alpha * MaxDrawdown_t0_to_k)
        Returns a series of the same length as df, where values are the SellScore relative to the
        most recent preceding buy signal.
        """
        exit_scores = pd.Series(0.0, index=df.index)
        buy_indices = signals[signals == 1].index.tolist()
        
        if not buy_indices:
            return exit_scores

        for i, t0_idx in enumerate(buy_indices):
            # Find the range until the next buy signal or end of dataframe
            end_idx = buy_indices[i+1] if i + 1 < len(buy_indices) else len(df)
            
            entry_price = df['Close'].loc[t0_idx]
            max_drawdown = 0.0
            
            # Calculate scores for all bars k from t0 + 1 to end_idx - 1
            for k in range(df.index.get_loc(t0_idx) + 1, df.index.get_loc(end_idx if i + 1 < len(buy_indices) else df.index[-1]) + 1):
                if k >= len(df): break
                
                current_price = df['Close'].iloc[k]
                time_elapsed = k - df.index.get_loc(t0_idx)
                gain_k = (current_price - entry_price) / entry_price
                
                # MaxDrawdown_t0_to_k (absolute value of worst open loss since entry)
                # We track the lowest 'Low' price seen so far in this trade relative to entry_price
                period_low = df['Low'].iloc[df.index.get_loc(t0_idx):k+1].min()
                drawdown = max(0.0, (entry_price - period_low) / entry_price)
                max_drawdown = max(max_drawdown, drawdown)
                
                score = (gain_k / (time_elapsed ** gamma)) - (alpha * max_drawdown)
                
                # Assign to the corresponding index
                exit_scores.iloc[k] = score

        return exit_scores
