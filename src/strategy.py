import json
import os
from typing import Dict, List, Any, Optional
import pandas as pd
from .signals.base import SignalModel, SignalEvent
from .signals.ml_models import MLSignalModel
from .signals.rule_based import DivergenceSignalModel

class Strategy:
    """
    Represents a trading strategy which consists of:
    1. A set of features (configuration).
    2. A set of models (signals) that use these features.
    
    Can be serialized/deserialized.
    """
    def __init__(self, name: str, models: List[SignalModel] = None, feature_config: Dict[str, Any] = None):
        self.name = name
        self.models = models or []
        self.feature_config = feature_config or {} # {feat_name: {params}}

    def add_model(self, model: SignalModel):
        # Prevent duplicates by name
        for m in self.models:
            if m.name == model.name:
                self.models.remove(m)
                break
        self.models.append(model)

    def generate_all_signals(self, df: pd.DataFrame, feature_data: Dict[str, pd.Series]) -> List[SignalEvent]:
        events = []
        for model in self.models:
            signals = model.generate_signals(df, feature_data)
            # Find non-zero signals
            signal_indices = signals[signals != 0].index
            for idx in signal_indices:
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
            "feature_config": self.feature_config,
            "models": [m.to_dict() for m in self.models]
        }

    def save(self, directory: str = "strategies"):
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, f"{self.name}.json")
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load(cls, name: str, directory: str = "strategies") -> 'Strategy':
        file_path = os.path.join(directory, f"{name}.json")
        if not os.path.exists(file_path):
            # Fallback to check if it exists in old "signal_models" directory just in case, or raise error
            old_path = os.path.join("signal_models", f"{name}.json")
            if os.path.exists(old_path):
                file_path = old_path
            else:
                raise FileNotFoundError(f"Strategy {name} not found.")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        models = []
        for m_data in data.get("models", []):
            m_type = m_data.get("type")
            if m_type == "MLSignalModel":
                m = MLSignalModel(m_data["name"], m_data.get("model_type", "RandomForest"))
                m.features_to_use = m_data.get("features_to_use", [])
                m.target_window = m_data.get("target_window", 5)
                m.target_threshold = m_data.get("target_threshold", 0.01)
                m.trained = m_data.get("trained", False)
                # Load other ML specific params if needed
            elif m_type == "DivergenceSignalModel":
                m = DivergenceSignalModel(m_data["name"], m_data.get("indicator", "RSI_14"))
                m.lookback = m_data.get("lookback", 20)
                m.order = m_data.get("order", 5)
            else:
                # Fallback or generic model loader
                print(f"Unknown signal model type: {m_type}")
                continue
            models.append(m)
            
        return cls(name=data["name"], models=models, feature_config=data.get("feature_config", {}))

    @staticmethod
    def list_available(directory: str = "strategies"):
        if not os.path.exists(directory):
            # Check old directory too just in case during migration
            if os.path.exists("signal_models"):
                 return [f.replace(".json", "") for f in os.listdir("signal_models") if f.endswith(".json")]
            return []
        return [f.replace(".json", "") for f in os.listdir(directory) if f.endswith(".json")]
