import json
import os
import pickle
import importlib.util
from typing import Dict, List, Any, Optional
import pandas as pd
from .signals.base import SignalEvent

SCRIPT_TEMPLATE = """import pandas as pd
import numpy as np
from src.signals.base import SignalModel

class StrategySignal(SignalModel):
    def generate_signals(self, df: pd.DataFrame, feature_data: dict) -> pd.Series:
        \"\"\"
        df: DataFrame with OHLCV data ('Open', 'High', 'Low', 'Close', 'Volume')
        feature_data: Dict of feature results (e.g. {'RSI_14': Series, ...})
        \"\"\"
        signals = pd.Series(0, index=df.index)
        
        # Example using OHLCV:
        # close = df['Close']
        # volume = df['Volume']
        
        # Example: 
        # 1 = Buy (Green Triangle Up)
        # -1 = Sell (Red Triangle Down)
        # 2 = Neutral (Yellow Circle)
        
        # if 'RSI_14' in feature_data:
        #     rsi = feature_data['RSI_14']
        #     signals[(rsi < 30)] = 1
        #     signals[(rsi > 70)] = -1
        #     signals[(rsi >= 45) & (rsi <= 55)] = 2
            
        return signals
"""

class Strategy:
    """
    Represents a trading strategy which consists of:
    1. A set of features (configuration).
    2. A Python script file that defines signal generation logic.
    
    Both are pickled into a single .strat file.
    """
    def __init__(self, name: str, feature_config: Dict[str, Any] = None, script_content: str = None, directory: str = "strategies"):
        self.name = name
        self.feature_config = feature_config or {} 
        self.script_content = script_content
        self.directory = directory
        self._ensure_script_exists()

    @property
    def script_path(self):
        # Working script path (extracted from pickle for execution/editing)
        return os.path.join(self.directory, "working_scripts", f"{self.name}.py")

    def _ensure_script_exists(self):
        os.makedirs(os.path.dirname(self.script_path), exist_ok=True)
        
        # If we have content (from a load), we write it to the working file ONLY if it doesn't exist
        if self.script_content is not None and not os.path.exists(self.script_path):
            with open(self.script_path, 'w') as f:
                f.write(self.script_content)
        # If no content and no file, use template
        elif not os.path.exists(self.script_path):
            with open(self.script_path, 'w') as f:
                f.write(SCRIPT_TEMPLATE)

    def generate_signals(self, df: pd.DataFrame, feature_data: Dict[str, pd.Series]) -> List[SignalEvent]:
        """
        Dynamically loads the StrategySignal class from the script file and executes it.
        Force reloads the module to pick up changes from VS Code.
        """
        if not os.path.exists(self.script_path):
            return []

        try:
            # Dynamic Import with unique name to avoid caching issues
            module_name = f"strategy_script_{self.name}"
            spec = importlib.util.spec_from_file_location(module_name, self.script_path)
            module = importlib.util.module_from_spec(spec)
            
            # Remove from sys.modules if it exists to force a fresh load
            import sys
            if module_name in sys.modules:
                del sys.modules[module_name]
                
            spec.loader.exec_module(module)
            
            if not hasattr(module, 'StrategySignal'):
                print(f"Error in '{self.script_path}': Class 'StrategySignal' not found.")
                return []
            
            model = module.StrategySignal()
            signals = model.generate_signals(df, feature_data)
            
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
        # Ensure script content is fresh from disk before serializing
        if os.path.exists(self.script_path):
            with open(self.script_path, 'r') as f:
                self.script_content = f.read()
        
        return {
            "name": self.name,
            "feature_config": self.feature_config,
            "script_content": self.script_content
        }

    def save(self):
        os.makedirs(self.directory, exist_ok=True)
        file_path = os.path.join(self.directory, f"{self.name}.strat")
        
        # Use pickle to bundle everything into one binary file
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
