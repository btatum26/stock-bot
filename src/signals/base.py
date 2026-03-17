from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import pandas as pd
from dataclasses import dataclass

@dataclass
class SignalEvent:
    name: str
    index: int
    timestamp: Any
    value: float
    side: str # 'buy', 'sell', or 'neutral'
    description: str

class SignalModel(ABC):
    """
    Base class for custom strategy signals.
    Users should inherit from this and implement generate_signals.
    """
    def __init__(self):
        self.name = self.__class__.__name__
        self.params = {} # Dynamic parameters injected from GUI

    @property
    def signal_parameters(self) -> Dict[str, Any]:
        """Override to provide strategy-specific parameters (rules)."""
        return {}

    @property
    def current_params(self) -> Dict[str, Any]:
        """Returns injected params if present, otherwise default signal_parameters."""
        if self.params:
            return self.params
        return self.signal_parameters

    @property
    def parameters(self) -> Dict[str, Any]:
        """Backward compatibility: alias for current_params."""
        return self.current_params

    @property
    def model_hyperparameters(self) -> Dict[str, Any]:
        """Override to provide ML training hyperparameters."""
        return {
            "n_estimators": 100,
            "max_depth": 10,
            "target_window": 5,
            "target_threshold": 0.01
        }

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame, feature_data: Dict[str, pd.Series]) -> pd.Series:
        """
        Processes feature data and returns a series of signals.
        1: Buy, -1: Sell, 0: Neutral.
        """
        pass
