from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import pandas as pd


# ---------------------------------------------------------------------------
# Visual output types — used by the GUI to render indicators
# ---------------------------------------------------------------------------

@dataclass
class FeatureOutput:
    """Base class for all visual overlay outputs."""
    name: str


@dataclass
class LineOutput(FeatureOutput):
    """A continuous line overlay on the chart."""
    data: List[Any]
    color: str = "#ffffff"
    width: float = 1.5
    schema_name: str = ""


@dataclass
class LevelOutput(FeatureOutput):
    """A horizontal price level (support/resistance zone)."""
    price: float = 0.0
    min_price: float = 0.0
    max_price: float = 0.0
    strength: float = 1.0
    color: str = "#0000ff"
    schema_name: str = ""


@dataclass
class MarkerOutput(FeatureOutput):
    """Scatter markers on the chart (e.g. divergence signals)."""
    indices: List[int] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    color: str = "#ffffff"
    shape: str = "o"
    schema_name: str = ""


@dataclass
class HeatmapOutput(FeatureOutput):
    """A price-density heatmap overlay."""
    price_grid: List[float] = field(default_factory=list)
    density: List[float] = field(default_factory=list)
    color_map: str = "blue"


@dataclass
class FeatureResult:
    """
    Full output of a feature computation.

    visuals: list of FeatureOutput subclass instances for the GUI to render.
    data:    raw Series keyed by column name, for signal/ML use.
    """
    visuals: List[FeatureOutput] = field(default_factory=list)
    data: Dict[str, pd.Series] = field(default_factory=dict)


# --- Base Feature Class ---
class Feature(ABC):
    """
    Abstract Base Class for all Stock Bot Features.
    """
    
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
        return {}

    @abstractmethod
    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> FeatureResult:
        """
        Main logic. Receives OHLCV DataFrame and current parameters.
        Returns a FeatureResult containing raw data.
        """
        pass
