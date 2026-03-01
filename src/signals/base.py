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

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame, feature_data: Dict[str, pd.Series]) -> pd.Series:
        """
        Processes feature data and returns a series of signals.
        1: Buy, -1: Sell, 0: Neutral.
        """
        pass
