import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, List, Any, Optional
from .base import SignalModel

class MLSignalModel(SignalModel):
    """
    A signal model that uses machine learning to produce buy/sell signals.
    """
    def __init__(self, name: str, model_type: str = "RandomForest"):
        super().__init__(name)
        self.model_type = model_type
        self.model = self._init_model()
        self.trained = False
        self.features_to_use: List[str] = []
        self.target_window: int = 5 # Predict if price goes up/down in 5 periods
        self.target_threshold: float = 0.01 # 1% price change for a signal

    def _init_model(self):
        if self.model_type == "RandomForest":
            return RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def prepare_data(self, df: pd.DataFrame, feature_data: Dict[str, pd.Series], label: bool = True):
        """
        Combines selected features into a single DataFrame for ML.
        If label=True, it also generates the target labels based on future price change.
        """
        # Align all features with the price dataframe
        X = pd.DataFrame(index=df.index)
        for f in self.features_to_use:
            if f in feature_data:
                X[f] = feature_data[f]
            elif f in df.columns:
                X[f] = df[f]
        
        # Drop rows where any selected feature is NaN (e.g., during SMA startup)
        X = X.dropna()
        y = None
        
        if label:
            # Create target: 1 if future price > threshold, -1 if future price < -threshold, 0 otherwise
            future_close = df['Close'].shift(-self.target_window)
            price_change = (future_close - df['Close']) / df['Close']
            
            y = pd.Series(0, index=df.index)
            y[price_change > self.target_threshold] = 1
            y[price_change < -self.target_threshold] = -1
            
            # Re-align X and y
            common_idx = X.index.intersection(y.index).dropna()
            # Remove the last target_window rows since we don't have future data for them
            common_idx = common_idx[:-self.target_window]
            
            X = X.loc[common_idx]
            y = y.loc[common_idx]
            
        return X, y

    def train(self, df: pd.DataFrame, feature_data: Dict[str, pd.Series]):
        """
        Trains the underlying ML model.
        """
        X, y = self.prepare_data(df, feature_data, label=True)
        if len(X) < 10:
            print("Not enough data to train model.")
            return False
        
        self.model.fit(X, y)
        self.trained = True
        return True

    def generate_signals(self, df: pd.DataFrame, feature_data: Dict[str, pd.Series]) -> pd.Series:
        """
        Uses the trained model to predict signals.
        """
        if not self.trained:
            return pd.Series(0, index=df.index)
        
        X, _ = self.prepare_data(df, feature_data, label=False)
        if X.empty:
            return pd.Series(0, index=df.index)
        
        predictions = self.model.predict(X)
        signals = pd.Series(0, index=df.index)
        signals.loc[X.index] = predictions
        return signals

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "model_type": self.model_type,
            "features_to_use": self.features_to_use,
            "target_window": self.target_window,
            "target_threshold": self.target_threshold,
            "trained": self.trained
        })
        return d
