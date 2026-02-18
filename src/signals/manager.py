import json
import os
from typing import Dict, List, Any, Optional
import pandas as pd
from .base import SignalModel, SignalEvent
from .ml_models import MLSignalModel
from .rule_based import DivergenceSignalModel

class SignalManager:
    """
    Manages loading, saving, and executing signal models.
    """
    def __init__(self, name: str, models: List[SignalModel] = None):
        self.name = name
        self.models = models or []

    def add_model(self, model: SignalModel):
        self.models.append(model)

    def generate_all_signals(self, df: pd.DataFrame, feature_data: Dict[str, pd.Series]) -> List[SignalEvent]:
        events = []
        for model in self.models:
            signals = model.generate_signals(df, feature_data)
            for idx in signals[signals != 0].index:
                iloc = df.index.get_loc(idx)
                side = 'buy' if signals[idx] == 1 else 'sell'
                events.append(SignalEvent(
                    name=model.name,
                    index=iloc,
                    timestamp=idx,
                    value=df['Close'].iloc[iloc],
                    side=side,
                    description=f"{model.name} Signal Generated"
                ))
        return events

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "models": [m.to_dict() for m in self.models]
        }

    def save(self, directory: str = "signal_models"):
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, f"{self.name}.json")
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load(cls, name: str, directory: str = "signal_models") -> 'SignalManager':
        file_path = os.path.join(directory, f"{name}.json")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Signal set {name} not found.")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
            models = []
            for m_data in data["models"]:
                m_type = m_data.get("type")
                if m_type == "MLSignalModel":
                    m = MLSignalModel(m_data["name"], m_data.get("model_type", "RandomForest"))
                    m.features_to_use = m_data.get("features_to_use", [])
                    m.target_window = m_data.get("target_window", 5)
                    m.target_threshold = m_data.get("target_threshold", 0.01)
                    m.trained = m_data.get("trained", False)
                    # We would need to load the actual ML model file here if it were saved separately.
                elif m_type == "DivergenceSignalModel":
                    m = DivergenceSignalModel(m_data["name"], m_data.get("indicator", "RSI_14"))
                    m.lookback = m_data.get("lookback", 20)
                    m.order = m_data.get("order", 5)
                else:
                    print(f"Unknown signal model type: {m_type}")
                    continue
                models.append(m)
            return cls(name=data["name"], models=models)

    @staticmethod
    def list_available(directory: str = "signal_models"):
        if not os.path.exists(directory):
            return []
        return [f.replace(".json", "") for f in os.listdir(directory) if f.endswith(".json")]
