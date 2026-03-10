# --- AVAILABLE FEATURES ---
# This strategy has access to the following features in `feature_data`:
# - Volume (Always available)
# - Support & Resistance
# - RSI
# - ATR
# - Moving Average
# --------------------------

import pandas as pd
import numpy as np
import xgboost as xgb
from src.signals.base import SignalModel
from typing import Dict, Any

class StrategySignal(SignalModel):
    def __init__(self):
        super().__init__()
        self.active_model_id = None
        self.model_instances = {}

    @property
    def parameters(self) -> Dict[str, Any]:
        """
        Define parameters that will show up in the Training GUI.
        """
        return {
            "drawdown_penalty": 2.0,
            "target_window": 20,
            "oversold_threshold": 30,
            "strength_threshold": 0.015,
            "n_estimators": 100,
            "max_depth": 5
        }

    def normalize_to_bar(self, df: pd.DataFrame, feature_data: dict, ref_idx: int) -> dict:
        """
        Normalizes price and specific technical indicators to the value at ref_idx.
        Converts them to percentage change relative to that bar.
        Does NOT normalize RSI.
        """
        normalized = {}
        
        # Reference prices
        ref_price = df['Close'].iloc[ref_idx]
        if ref_price == 0: ref_price = 1e-9
        
        # Normalize OHLC Price Data
        normalized['Close'] = (df['Close'] - ref_price) / ref_price
        normalized['High'] = (df['High'] - ref_price) / ref_price
        normalized['Low'] = (df['Low'] - ref_price) / ref_price
        normalized['Open'] = (df['Open'] - ref_price) / ref_price

        # Normalize Features
        for key, series in feature_data.items():
            if 'RSI' in key:
                normalized[key] = series
            else:
                ref_val = series.iloc[ref_idx]
                if pd.isna(ref_val) or ref_val == 0:
                    normalized[key] = (series - series.mean()) / (series.std() if series.std() != 0 else 1.0)
                else:
                    normalized[key] = (series - ref_val) / ref_val

        return normalized

    def volatility_scale_zscore(self, price_series: pd.Series, n: int = 20) -> pd.Series:
        """Z-Score Method (Statistical Normalization)."""
        rolling_mean = price_series.rolling(window=n).mean()
        rolling_std = price_series.rolling(window=n).std()
        return (price_series - rolling_mean) / rolling_std

    def volatility_scale_atr(self, price_series: pd.Series, atr_series: pd.Series) -> pd.Series:
        """ATR Method (Technical Trading)."""
        price_diff = price_series.diff()
        return price_diff / atr_series

    def volatility_scale_returns(self, price_series: pd.Series, n: int = 20) -> pd.Series:
        """Volatility-Scaled Returns (Machine Learning & Quant Models)."""
        adj_price = price_series + 1.0 if price_series.max() < 10.0 else price_series
        log_returns = pd.Series(np.log(adj_price / adj_price.shift(1)), index=price_series.index)
        rolling_vol = log_returns.rolling(window=n).std()
        return log_returns / rolling_vol

    def generate_raw_buy_signals(self, df: pd.DataFrame, feature_data: dict, settings: dict = None) -> list:
        """
        Identifies all potential buy signals based on the core RSI reversal logic.
        Returns a list of indices where a signal is triggered.
        """
        rsi = next((v for k, v in feature_data.items() if 'RSI' in k), None)
        if rsi is None: return []

        indices = []
        
        # Use provided settings or defaults from self.parameters
        if settings is None:
            settings = self.parameters
        
        oversold_threshold = settings.get("oversold_threshold", 30)
        
        # Look for Bullish RSI Reversal
        for i in range(20, len(df)):
            is_oversold = rsi.iloc[i-1] < oversold_threshold
            is_reversal = rsi.iloc[i] > rsi.iloc[i-1] and rsi.iloc[i-1] <= rsi.iloc[i-2]
            
            if is_oversold and is_reversal:
                indices.append(i)
        
        return indices

    def train(self, df: pd.DataFrame, feature_data: dict, settings: dict):
        """
        Train an XGBoost model to predict a "Potential Score" for buy signals.
        Score = (MaxGain / TimeToMaxGain) - (alpha * MaxDrawdown)
        """
        raw_indices = self.generate_raw_buy_signals(df, feature_data, settings)
        
        target_window = int(settings.get("target_window", 20))
        alpha = float(settings.get("drawdown_penalty", 2.0))
        candidates = [idx for idx in raw_indices if idx < len(df) - target_window]

        if len(candidates) < 10:
            raise ValueError(f"Not enough signal candidates to train. Found {len(candidates)}.")

        atr = next((v for k, v in feature_data.items() if 'ATR' in k), None)
        rsi = next((v for k, v in feature_data.items() if 'RSI' in k), None)

        X_list = []
        y_list = []

        zscore = self.volatility_scale_zscore(df['Close'])
        vol_returns = self.volatility_scale_returns(df['Close'])
        atr_scale = self.volatility_scale_atr(df['Close'], atr) if atr is not None else None

        for idx in candidates:
            # --- FEATURE CALCULATION ---
            norm = self.normalize_to_bar(df, feature_data, idx)
            feat = {
                "RSI": rsi.iloc[idx],
                "ZScore": zscore.iloc[idx],
                "VolReturns": vol_returns.iloc[idx],
                "ATR_Scale": atr_scale.iloc[idx] if atr_scale is not None else 0
            }
            for k, v in norm.items():
                feat[f"Norm_{k}"] = v.iloc[idx]
            X_list.append(feat)
            
            # --- TARGET CALCULATION (Score_t) ---
            entry_price = df['Close'].iloc[idx]
            future_prices = df['Close'].iloc[idx+1 : idx+1+target_window]
            
            # Max Gain and Time to Max Gain
            max_price = future_prices.max()
            max_idx = future_prices.idxmax()
            
            max_gain = (max_price - entry_price) / entry_price
            time_to_max = (max_idx - idx) # Number of bars
            if time_to_max <= 0: time_to_max = 1
            
            # Max Drawdown experienced BEFORE hitting the max gain
            prices_before_peak = df['Close'].iloc[idx+1 : max_idx + 1]
            if not prices_before_peak.empty:
                min_price_before_peak = prices_before_peak.min()
                max_drawdown = (entry_price - min_price_before_peak) / entry_price
                if max_drawdown < 0: max_drawdown = 0 # Price never went below entry
            else:
                max_drawdown = 0

            # Composite Score
            score = (max_gain / time_to_max) - (alpha * max_drawdown)
            y_list.append(score)

        X = pd.DataFrame(X_list)
        y = pd.Series(y_list)

        model = xgb.XGBRegressor(
            n_estimators=int(settings.get("n_estimators", 100)),
            max_depth=int(settings.get("max_depth", 5)),
            learning_rate=0.1,
            objective='reg:squarederror'
        )
        model.fit(X, y)

        metrics = {"samples": len(candidates), "avg_score": float(y.mean())}
        return model, metrics

    def generate_signals(self, df: pd.DataFrame, feature_data: dict) -> pd.Series:
        """
        Entry: RSI Bullish Reversal from Oversold + XGBoost Ranking.
        """
        signals = pd.Series(0, index=df.index)
        
        if self.active_model_id not in self.model_instances:
            return signals

        model_info = self.model_instances[self.active_model_id]
        model = model_info['weights']
        settings = model_info.get('settings', self.parameters)

        # Get raw signal candidates
        raw_indices = self.generate_raw_buy_signals(df, feature_data, settings)
        if not raw_indices:
            return signals

        atr = next((v for k, v in feature_data.items() if 'ATR' in k), None)
        rsi = next((v for k, v in feature_data.items() if 'RSI' in k), None)
        
        # Pre-calculate global volatility features
        zscore = self.volatility_scale_zscore(df['Close'])
        vol_returns = self.volatility_scale_returns(df['Close'])
        atr_scale = self.volatility_scale_atr(df['Close'], atr) if atr is not None else None

        strength_threshold = float(settings.get("strength_threshold", 0.015))

        for i in raw_indices:
            # Calculate features for this specific bar
            norm = self.normalize_to_bar(df, feature_data, i)
            feat = {
                "RSI": rsi.iloc[i],
                "ZScore": zscore.iloc[i],
                "VolReturns": vol_returns.iloc[i],
                "ATR_Scale": atr_scale.iloc[i] if atr_scale is not None else 0
            }
            for k, v in norm.items():
                feat[f"Norm_{k}"] = v.iloc[i]
            
            X_bar = pd.DataFrame([feat])
            
            # Ensure feature order matches training
            if hasattr(model, 'feature_names_in_'):
                X_bar = X_bar[model.feature_names_in_]
            
            prediction = model.predict(X_bar)[0]
            
            # Trigger if prediction is strong enough
            if prediction >= strength_threshold:
                signals.iloc[i] = 1

        return signals
