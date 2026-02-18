from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import pandas as pd

# --- Output Types ---
@dataclass
class FeatureOutput:
    name: str
    color: str = 'white'

@dataclass
class LineOutput(FeatureOutput):
    data: List[float] = None # Matches length of DF
    width: int = 1

@dataclass
class LevelOutput(FeatureOutput):
    price: float = 0.0
    min_price: float = 0.0
    max_price: float = 0.0
    strength: float = 0.0

@dataclass
class MarkerOutput(FeatureOutput):
    indices: List[int] = None
    values: List[float] = None
    shape: str = 'o' # 'o', 't', 's', 'd', '+', 'x'

@dataclass
class HeatmapOutput(FeatureOutput):
    """
    Represents a vertical density gradient (e.g. for KDE).
    price_grid: Array of price points (Y-axis)
    density: Array of density values (0.0 to 1.0) for each price point.
    """
    price_grid: List[float] = None
    density: List[float] = None
    color_map: str = 'plasma' # matplotlib colormap name or similar

# --- Base Feature Class ---
class Feature(ABC):
    """
    Abstract Base Class for all Stock Bot Features.
    """
    
    @property
    def target_pane(self) -> str:
        """
        'main': Overlay on price chart.
        'new': Create a new subplot below.
        """
        return "main"

    @property
    @abstractmethod
    def name(self) -> str:
        """Display name of the feature."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Short description of what the feature does."""
        pass

    @property
    @abstractmethod
    def category(self) -> str:
        """Category of the feature (e.g., 'Price Levels', 'Trend', 'Volume')."""
        pass

    @property
    def parameters(self) -> Dict[str, Any]:
        """
        Dictionary of default parameters.
        Example: {'window': 14, 'threshold': 0.01}
        """
        pass

    @property
    def y_range(self) -> Optional[List[float]]:
        """
        Fixed Y-axis range [min, max].
        None if dynamic/data-driven.
        """
        return None

    @property
    def y_padding(self) -> float:
        """
        Default Y-axis padding (0.1 = 10% of data height).
        """
        return 0.1

    @abstractmethod
    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> List[FeatureOutput]:
        """
        Main logic. Receives OHLCV DataFrame and current parameters.
        Returns a list of FeatureOutput objects (Lines, Levels, Markers).
        """
        pass
