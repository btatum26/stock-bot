import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from src.signals.base import SignalModel

class StrategySignal(SignalModel):
    def __init__(self):
        super().__init__()
        self.active_model_id = None
        self.model_instances = {}

    def train(self, df: pd.DataFrame, feature_data: dict, settings: dict):
        """
        Train an exit model: predicts if the price will drop (Sell) 
        based on the provided technical indicators (e.g., RSI, ATR).
        """
        # 1. Prepare Features (X)
        X = pd.DataFrame(index=df.index)
        for k, v in feature_data.items():
            X[k] = v
        X = X.dropna()
        
        if X.empty:
            raise ValueError("No valid feature data available for training.")

        # 2. Prepare Labels (y) - Exit Logic (Predicting Drops)
        target_window = settings.get("target_window", 5)
        target_threshold = settings.get("target_threshold", 0.01)
        
        future_close = df['Close'].shift(-target_window)
        price_change = (future_close - df['Close']) / df['Close']
        
        y = pd.Series(0, index=df.index)
        # We only care about predicting drops (Exits) for this specific hybrid strategy
        y[price_change < -target_threshold] = -1 
        
        common_idx = X.index.intersection(y.index).dropna()[:-target_window]
        X, y = X.loc[common_idx], y.loc[common_idx]
        
        if len(X) < 20:
            raise ValueError(f"Not enough data to train. Need at least 20 samples, got {len(X)}.")
            
        # 3. Train/Test Split
        split = int(len(X) * (1 - settings.get('validation_split', 0.2)))
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        
        # 4. Train Model
        model_type = settings.get('model_type', 'RandomForest')
        if model_type == "RandomForest":
            model = RandomForestClassifier(
                n_estimators=settings.get('n_estimators', 100),
                max_depth=settings.get('max_depth', 10),
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        model.fit(X_train, y_train)
        
        # 5. Evaluate
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
            "samples": len(X)
        }
        
        return model, metrics

    def generate_signals(self, df: pd.DataFrame, feature_data: dict) -> pd.Series:
        """
        df: DataFrame with OHLCV data ('Open', 'High', 'Low', 'Close', 'Volume')
        feature_data: Dict of feature results (e.g. {'RSI_14': Series, 'ATR_14': Series, 'Volume': Series, 'Dist_to_Support': Series})
        """
        # --- ML Exit Logic Setup ---
        ml_signals = None
        if self.active_model_id and self.active_model_id in self.model_instances:
            # Get the raw scikit-learn model object
            model = self.model_instances[self.active_model_id]['weights']
            
            # Predict using ONLY the features the model was trained on
            X = pd.DataFrame(index=df.index)
            for k, v in feature_data.items(): X[k] = v
            
            # Check if model has information about its training features
            if hasattr(model, 'feature_names_in_'):
                missing = [f for f in model.feature_names_in_ if f not in X.columns]
                if missing:
                    # Model expects features that are not provided in feature_data
                    return signals
                
                # Filter to only the features expected by the model, in the same order
                X = X[model.feature_names_in_]
            
            X_clean = X.dropna()
            
            if not X_clean.empty:
                preds = model.predict(X_clean)
                ml_signals = pd.Series(0, index=df.index)
                ml_signals.loc[X_clean.index] = preds

        signals = pd.Series(0, index=df.index)
        
        # 1. Extract RSI for Entry Logic
        rsi = None
        for key in feature_data.keys():
            if key.startswith('RSI'): 
                rsi = feature_data[key]
                break
        
        if rsi is None:
            return signals

        # 2. Strategy Parameters
        oversold_threshold = 30
        
        # 3. State Management
        in_position = False
        
        for i in range(2, len(df)):
            if not in_position:
                # ENTRY: Bullish RSI Reversal (Hardcoded)
                is_oversold = rsi.iloc[i-1] < oversold_threshold
                is_reversal = rsi.iloc[i] > rsi.iloc[i-1] and rsi.iloc[i-1] <= rsi.iloc[i-2]
                
                if is_oversold and is_reversal:
                    signals.iloc[i] = 1 # BUY
                    in_position = True
            else:
                # EXIT LOGIC: Exclusively ML-driven (predicts -1 for Sell)
                should_exit = False

                if ml_signals is not None and ml_signals.iloc[i] == -1:
                    should_exit = True

                if should_exit:
                    signals.iloc[i] = -1 # SELL
                    in_position = False
                    
        return signals
