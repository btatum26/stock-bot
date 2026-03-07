import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score
from typing import Dict, List, Any, Optional
from .base import SignalModel

class MLSignalModel(SignalModel):
    """
    A signal model that uses machine learning to produce buy/sell signals.
    This class now acts as a manager/factory for model instances.
    """
    def __init__(self, name: str, model_type: str = "RandomForest"):
        super().__init__(name)
        self.model_type = model_type
        self.features_to_use: List[str] = []
        self.target_window: int = 5 
        self.target_threshold: float = 0.01 

    def _init_model(self, settings: Dict[str, Any]):
        if self.model_type == "RandomForest":
            return RandomForestClassifier(
                n_estimators=settings.get("n_estimators", 100), 
                max_depth=settings.get("max_depth", 10), 
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def prepare_data(self, df: pd.DataFrame, feature_data: Dict[str, pd.Series], label: bool = True):
        X = pd.DataFrame(index=df.index)
        for f in self.features_to_use:
            if f in feature_data:
                X[f] = feature_data[f]
            elif f in df.columns:
                X[f] = df[f]
        
        X = X.dropna()
        y = None
        
        if label:
            future_close = df['Close'].shift(-self.target_window)
            price_change = (future_close - df['Close']) / df['Close']
            
            y = pd.Series(0, index=df.index)
            y[price_change > self.target_threshold] = 1
            y[price_change < -self.target_threshold] = -1
            
            common_idx = X.index.intersection(y.index).dropna()
            common_idx = common_idx[:-self.target_window]
            
            X = X.loc[common_idx]
            y = y.loc[common_idx]
            
        return X, y

    def train(self, df: pd.DataFrame, feature_data: Dict[str, pd.Series], settings: Dict[str, Any], validation_split: float = 0.2, script_instance: Any = None):
        """
        Trains the model and returns (weights, metrics).
        """
        self.target_window = settings.get("target_window", 5)
        self.target_threshold = settings.get("target_threshold", 0.01)

        if script_instance and hasattr(script_instance, 'prepare_labels'):
            # Use custom labeling logic from the script if available
            y = script_instance.prepare_labels(df, feature_data, settings)
            X, _ = self.prepare_data(df, feature_data, label=False)
            
            # Re-align X and y
            common_idx = X.index.intersection(y.index).dropna()
            X = X.loc[common_idx]
            y = y.loc[common_idx]
        else:
            X, y = self.prepare_data(df, feature_data, label=True)

        if len(X) < 20:
            raise ValueError(f"Not enough data to train model. Need at least 20 samples, got {len(X)}.")
        
        # Simple train/test split
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        model = self._init_model(settings)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
            "samples": len(X)
        }
        
        return model, metrics

    def generate_signals(self, df: pd.DataFrame, feature_data: Dict[str, pd.Series], model_weights: Any) -> pd.Series:
        """
        Uses a specific trained model (weights) to predict signals.
        """
        if model_weights is None:
            return pd.Series(0, index=df.index)
        
        X, _ = self.prepare_data(df, feature_data, label=False)
        if X.empty:
            return pd.Series(0, index=df.index)
        
        predictions = model_weights.predict(X)
        signals = pd.Series(0, index=df.index)
        signals.loc[X.index] = predictions
        return signals
