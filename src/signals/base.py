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
    side: str # 'buy' or 'sell'
    description: str

class SignalModel(ABC):
    """
    Base class for models that generate signals from feature data.
    """
    def __init__(self, name: str):
        self.name = name
        self.features_required: List[str] = []

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame, feature_data: Dict[str, pd.Series]) -> pd.Series:
        """
        Processes feature data and returns a series of signals.
        1: Buy, -1: Sell, 0: Neutral.
        """
        pass

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "features_required": self.features_required
        }
