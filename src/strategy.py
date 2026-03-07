import json
import os
import pickle
import importlib.util
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime
from .signals.base import SignalEvent

SCRIPT_TEMPLATE = """import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
from sklearn.metrics import precision_score, recall_score, accuracy_score
from src.signals.base import SignalModel

class StrategySignal(SignalModel):
    def __init__(self):
        super().__init__()
        self.active_model_id = None
        self.model_instances = {} # UUID -> {weights, metrics, settings, ...}

    def train(self, df: pd.DataFrame, feature_data: dict, settings: dict):
        \"\"\"
        Self-contained training logic.
        Returns: (trained_model_object, metrics_dict)
        \"\"\"
        # 1. Prepare Data
        X = pd.DataFrame(index=df.index)
        for k, v in feature_data.items(): X[k] = v
        X = X.dropna()
        
        # 2. Prepare Labels
        target_window = settings.get("target_window", 5)
        target_threshold = settings.get("target_threshold", 0.01)
        
        # We need to compute the target on a per-ticker basis if we want to be precise,
        # but since 'df' here is already a concatenated long-format DF from TrainingThread,
        # we have to be careful about cross-ticker shifts. 
        # However, for simplicity in this template, we'll assume the concat was done safely.
        
        future_close = df['Close'].shift(-target_window)
        price_change = (future_close - df['Close']) / df['Close']
        
        y = pd.Series(0, index=df.index)
        y[price_change > target_threshold] = 1
        y[price_change < -target_threshold] = -1
        
        # Remove the last few rows where we don't have future data
        common_idx = X.index.intersection(y.index).dropna()[:-target_window]
        X, y = X.loc[common_idx], y.loc[common_idx]
        
        if len(X) < 20:
            raise ValueError(f"Not enough data to train (found {len(X)} samples).")
            
        # 3. Train/Test Split
        split = int(len(X) * (1 - settings.get('validation_split', 0.2)))
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        
        # 4. Train Model
        model_type = settings.get("model_type", "RandomForest")
        
        if model_type == "XGBoost" and XGBClassifier is not None:
            # XGBoost expects 0, 1, 2 for multiclass, so map [-1, 0, 1] -> [0, 1, 2]
            y_train_mapped = y_train + 1
            model = XGBClassifier(
                n_estimators=settings.get('n_estimators', 100),
                max_depth=settings.get('max_depth', 6),
                random_state=42,
                use_label_encoder=False,
                eval_metric='mlogloss'
            )
            model.fit(X_train, y_train_mapped)
            # Add a wrapper for predict to map back to [-1, 0, 1]
            original_predict = model.predict
            model.predict = lambda x: original_predict(x) - 1
        else:
            model = RandomForestClassifier(
                n_estimators=settings.get('n_estimators', 100),
                max_depth=settings.get('max_depth', 10),
                random_state=42
            )
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
        signals = pd.Series(0, index=df.index)
        
        # --- Using the active trained model ---
        if self.active_model_id and self.active_model_id in self.model_instances:
            model = self.model_instances[self.active_model_id]['weights']
            
            X = pd.DataFrame(index=df.index)
            for k, v in feature_data.items(): X[k] = v
            X_clean = X.dropna()
            
            if not X_clean.empty:
                preds = model.predict(X_clean)
                signals.loc[X_clean.index] = preds
                
        return signals
"""

class Strategy:
    """
    Represents a trading strategy which consists of:
    1. A set of features (configuration).
    2. A Python script file that defines signal generation logic.
    3. Trained ML model instances and metadata.
    
    Everything is pickled into a single .strat file.
    """
    def __init__(self, name: str, 
                 feature_config: Dict[str, Any] = None, 
                 script_content: str = None, 
                 model_instances: Dict[str, Any] = None,
                 model_definitions: Dict[str, Any] = None,
                 active_model_id: str = None,
                 metadata: Dict[str, Any] = None,
                 directory: str = "strategies"):
        self.name = name
        self.feature_config = feature_config or {} 
        self.script_content = script_content
        self.model_instances = model_instances or {} # UUID -> {weights, timestamp, comment, settings, training_scope, metrics}
        self.model_definitions = model_definitions or {
            "RandomForest": {"n_estimators": 100, "max_depth": 10},
            "XGBoost": {"n_estimators": 100, "max_depth": 6} 
        }
        self.active_model_id = active_model_id
        self.metadata = metadata or {
            "author": "System",
            "version": "1.0",
            "creation_date": datetime.now().isoformat()
        }
        self.directory = directory
        self._ensure_script_exists()

    @property
    def script_path(self):
        return os.path.join(self.directory, "working_scripts", f"{self.name}.py")

    def _ensure_script_exists(self):
        os.makedirs(os.path.dirname(self.script_path), exist_ok=True)
        if self.script_content is not None and not os.path.exists(self.script_path):
            with open(self.script_path, 'w') as f:
                f.write(self.script_content)
        elif not os.path.exists(self.script_path):
            with open(self.script_path, 'w') as f:
                f.write(SCRIPT_TEMPLATE)

    def get_script_instance(self) -> Optional[Any]:
        """
        Dynamically loads and instantiates the StrategySignal class from the script.
        """
        if not os.path.exists(self.script_path):
            return None

        try:
            module_name = f"strategy_script_{self.name}"
            spec = importlib.util.spec_from_file_location(module_name, self.script_path)
            module = importlib.util.module_from_spec(spec)
            
            import sys
            if module_name in sys.modules:
                del sys.modules[module_name]
                
            spec.loader.exec_module(module)
            
            if hasattr(module, 'StrategySignal'):
                return module.StrategySignal()
        except Exception as e:
            print(f"Error loading script instance: {e}")
        return None

    def generate_signals(self, df: pd.DataFrame, feature_data: Dict[str, pd.Series]) -> List[SignalEvent]:
        if not os.path.exists(self.script_path):
            return []

        try:
            model_instance = self.get_script_instance()
            if not model_instance:
                return []
            
            # Pass the raw instances directly to the script
            model_instance.model_instances = self.model_instances
            model_instance.active_model_id = self.active_model_id
            
            signals = model_instance.generate_signals(df, feature_data)
            
            # Convert Series to Events
            events = []
            signal_indices = signals[signals != 0].index
            for idx in signal_indices:
                iloc = df.index.get_loc(idx)
                val = signals[idx]
                
                side = 'buy' if val == 1 else 'sell' if val == -1 else 'neutral' if val == 2 else 'unknown'
                if side == 'unknown': continue
                
                events.append(SignalEvent(
                    name=f"{self.name}_Signal",
                    index=iloc,
                    timestamp=idx,
                    value=df['Close'].iloc[iloc],
                    side=side,
                    description=f"Strategy Signal ({side})"
                ))
            return events
        except Exception as e:
            print(f"Error executing signal script for '{self.name}': {e}")
            import traceback
            traceback.print_exc()
            return []

    def to_dict(self) -> Dict[str, Any]:
        if os.path.exists(self.script_path):
            with open(self.script_path, 'r') as f:
                self.script_content = f.read()
        
        return {
            "name": self.name,
            "feature_config": self.feature_config,
            "script_content": self.script_content,
            "model_instances": self.model_instances,
            "model_definitions": self.model_definitions,
            "active_model_id": self.active_model_id,
            "metadata": self.metadata
        }

    def save(self):
        os.makedirs(self.directory, exist_ok=True)
        file_path = os.path.join(self.directory, f"{self.name}.strat")
        with open(file_path, 'wb') as f:
            pickle.dump(self.to_dict(), f)

    @classmethod
    def load(cls, name: str, directory: str = "strategies") -> 'Strategy':
        file_path = os.path.join(directory, f"{name}.strat")
        if not os.path.exists(file_path):
            return cls(name=name, directory=directory)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return cls(
            name=data["name"], 
            feature_config=data.get("feature_config", {}),
            script_content=data.get("script_content"),
            model_instances=data.get("model_instances", {}),
            model_definitions=data.get("model_definitions", {}),
            active_model_id=data.get("active_model_id"),
            metadata=data.get("metadata"),
            directory=directory
        )

    @staticmethod
    def list_available(directory: str = "strategies"):
        if not os.path.exists(directory):
            return []
        names = []
        for f in os.listdir(directory):
            if f.endswith(".strat"):
                strat_name = f.replace(".strat", "")
                if strat_name != "Default":
                    names.append(strat_name)
        return sorted(names)
