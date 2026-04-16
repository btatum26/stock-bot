"""Base classes and data structures for the quantitative feature engineering system.

This module defines the core contracts, output structures, and global registry
used to build and compute financial indicators or features consistently.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Type, TYPE_CHECKING
import pandas as pd

if TYPE_CHECKING:
    from .features import FeatureCache

# ---------------------------------------------------------------------------
# Global Registry
# ---------------------------------------------------------------------------
FEATURE_REGISTRY: Dict[str, Type['Feature']] = {}

def register_feature(name: str):
    """Registers a Feature class into the global system registry.

    This decorator allows feature classes to be dynamically discovered and
    instantiated by the orchestrator using their string identifier.

    Args:
        name (str): The unique identifier for the feature (e.g., "RSI", "MACD").

    Returns:
        Callable: The decorated class, now accessible via the global registry.
    """
    def decorator(cls: Type['Feature']):
        FEATURE_REGISTRY[name] = cls
        return cls
    return decorator


# ---------------------------------------------------------------------------
# Output Type System
# ---------------------------------------------------------------------------
class OutputType(str, Enum):
    """The 7 structural data shapes a feature can produce.

    These describe data structure only — no colors, widths, or rendering info.
    The GUI interprets these shapes and decides how to draw them.
    """
    LINE      = "line"       # pd.Series — one value per bar
    LEVEL     = "level"      # list of {value, label, strength?}
    BAND      = "band"       # references two line outputs (upper/lower)
    HISTOGRAM = "histogram"  # pd.Series — signed magnitude per bar
    MARKER    = "marker"     # pd.Series — sparse, NaN-gapped events
    ZONE      = "zone"       # list of {start, end, upper, lower, label?}
    HEATMAP   = "heatmap"    # {price_grid, time_index, intensity}


class Pane(str, Enum):
    """Where the output lives structurally.

    OVERLAY: data is in the price coordinate space (e.g., moving average).
    NEW:     data has its own y-axis range (e.g., RSI 0-100).
    """
    OVERLAY = "overlay"
    NEW     = "new"


@dataclass(frozen=True)
class OutputSchema:
    """Declares one logical output of a feature — its name, data shape, and pane.

    This is structural metadata, not rendering config. A distributed worker
    uses this to organize data; the GUI uses it to pick a renderer.

    Attributes:
        name:        Suffix or label (e.g., "upper", "mid"). None for single-output.
        output_type: One of the 7 data shapes.
        pane:        Where this output lives (overlay on price chart, or new sub-pane).
        band_pair:   For BAND type only — tuple of (upper_suffix, lower_suffix)
                     referencing two LINE outputs that form the band boundaries.
        y_range:     Optional fixed y-axis bounds (e.g., [0, 100] for RSI).
    """
    name: Optional[str]
    output_type: OutputType
    pane: Pane = Pane.OVERLAY
    band_pair: Optional[tuple] = None
    y_range: Optional[tuple] = None


# ---------------------------------------------------------------------------
# Feature Result
# ---------------------------------------------------------------------------
@dataclass
class FeatureResult:
    """Payload returned by a Feature's compute method.

    Holds all 7 data shapes. Most features only populate `data`.
    The GUI reads whichever fields are non-empty.

    Attributes:
        data:     Time-series outputs (lines, histograms, markers) keyed by column name.
        levels:   Horizontal price thresholds [{value, label, strength?}, ...].
        zones:    Rectangular price-time regions [{start, end, upper, lower, label?}, ...].
        heatmaps: 2D intensity grids keyed by name {name: {price_grid, time_index, intensity}}.
    """
    data: Dict[str, pd.Series] = None
    levels: List[Dict[str, Any]] = None
    zones: List[Dict[str, Any]] = None
    heatmaps: Dict[str, Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Base Feature Class
# ---------------------------------------------------------------------------
class Feature(ABC):
    """Abstract Base Class for quantitative indicators and features.

    This class enforces strict contracts for feature generation, ensuring
    vectorized computation, memory safety, and standardized output formats
    across the entire system.
    """

    @staticmethod
    def generate_column_name(feature_id: str, params: Dict[str, Any], output_name: Optional[str] = None) -> str:
        """Standardizes pandas DataFrame column naming to ensure consistency.

        Args:
            feature_id (str): The base name of the feature.
            params (Dict[str, Any]): The hyperparameter dictionary used for calculation.
            output_name (Optional[str], optional): Suffix for multi-output features.
                Defaults to None.

        Returns:
            str: The deterministic column name based on parameters.
        """
        norm = params.get("normalize", "none")
        prefix = "Norm_" if norm != "none" else ""

        ignored_keys = ["color", "normalize", "overbought", "oversold"]
        core_params = {k: v for k, v in params.items() if k not in ignored_keys and not k.startswith("color_")}

        if "type" in core_params and str(core_params["type"]).upper() == feature_id.upper():
            del core_params["type"]

        if len(core_params) == 1 and ("period" in core_params or "window" in core_params):
            val = core_params.get("period") or core_params.get("window")
            base = f"{feature_id}_{val}"
        elif not core_params:
            base = feature_id
        else:
            param_str = "_".join([f"{v}" for k, v in sorted(core_params.items())])
            base = f"{feature_id}_{param_str}"

        suffix = f"_{output_name.upper()}" if output_name else ""
        return f"{prefix}{base}{suffix}"

    # ------------------------------------------------------------------
    # Output schema — override per feature to declare output shapes
    # ------------------------------------------------------------------

    @property
    def output_schema(self) -> List[OutputSchema]:
        """Declares the structural outputs this feature produces.

        Each entry describes one logical output: its name, data shape, and pane.
        The GUI uses this to pick a renderer per output. The engine uses it to
        organize data for strategies and the ML bridge.

        Override this in subclasses. The default assumes a single line on a new pane.

        Returns:
            List[OutputSchema]: One entry per logical output.
        """
        return [OutputSchema(name=None, output_type=OutputType.LINE, pane=Pane.NEW)]

    @property
    def outputs(self) -> List[Optional[str]]:
        """Defines the raw data column suffixes produced by this feature.

        Derived from output_schema for backward compatibility with the
        orchestrator and workspace manager.

        Returns:
            List[Optional[str]]: A list of string suffixes, or a list containing
                None if the feature produces a single primary output.
        """
        return [s.name for s in self.output_schema if s.output_type in (
            OutputType.LINE, OutputType.HISTOGRAM, OutputType.MARKER
        )]

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the display name of the feature."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Returns a brief description of the feature's logic."""
        pass

    @property
    @abstractmethod
    def category(self) -> str:
        """Returns the categorization grouping of the feature."""
        pass

    @property
    def parameters(self) -> Dict[str, Any]:
        """Provides the default parameters required for computation.

        Returns:
            Dict[str, Any]: A dictionary of parameter keys and default values.
        """
        return {}

    @property
    def parameter_options(self) -> Dict[str, Dict[str, Any]]:
        """Provides metadata defining the bounds and types of parameters.

        Returns:
            Dict[str, Dict[str, Any]]: Configuration dictionaries for each parameter.
        """
        return {}

    def non_stationary_outputs(self, params: Dict[str, Any]) -> List[str]:
        """Declares which of this feature's output columns are non-stationary.

        Non-stationary outputs (those that track raw price levels or accumulate
        monotonically — moving averages, OBV, raw ATR, price-level S/R) should
        be passed through fractional differentiation before the ML scaler so
        the train/test distributions remain compatible.

        The default is an empty list — the feature's outputs are bounded or
        already stationary (RSI, ADX, ROC, pct_distance-normalized MA, etc.).
        Override in subclasses that emit unbounded price-coupled series.

        Args:
            params: The parameter dict that will be passed to ``compute``.
                Some features are stationary only under certain parameter
                values (e.g. a MovingAverage with ``normalize="pct_distance"``
                is stationary; one with ``normalize="none"`` is not).

        Returns:
            List of full column names (as produced by ``generate_column_name``)
            that the trainer should route through FFD instead of MinMax.
        """
        return []

    def normalize(self, df: pd.DataFrame, series: pd.Series, method: str, window: int = 20) -> pd.Series:
        """Systematically normalizes raw indicator data for downstream machine learning.

        Args:
            df (pd.DataFrame): The base market dataset containing price information.
            series (pd.Series): The raw computed indicator series.
            method (str): The string identifier for the normalization strategy
                ('pct_distance', 'price_ratio', 'z_score', or 'none').
            window (int, optional): The rolling window used for z-score calculations.
                Defaults to 20.

        Returns:
            pd.Series: The normalized data series.

        Raises:
            ValueError: If an unsupported normalization method is requested.
        """
        if method == "none" or not method:
            return series

        close = df['Close'] if 'Close' in df.columns else df['close']

        if method == "pct_distance":
            return (close - series) / series.replace(0, 1e-9)

        elif method == "price_ratio":
            return series / close.replace(0, 1e-9)

        elif method == "z_score":
            rolling_mean = series.rolling(window=window).mean()
            rolling_std = series.rolling(window=window).std().replace(0, 1e-9)
            return (series - rolling_mean) / rolling_std

        else:
            raise ValueError(f"Unknown normalization method: {method}")

    @abstractmethod
    def compute(self, df: pd.DataFrame, params: Dict[str, Any], cache: Optional['FeatureCache'] = None) -> FeatureResult:
        """Executes the core mathematical logic for the feature.

        This method must be strictly vectorized. It receives the raw price data
        and is responsible for calculating the indicator series without mutating
        the input DataFrame. Intermediate computations should leverage the cache.

        Args:
            df (pd.DataFrame): The raw market data containing OHLCV columns.
            params (Dict[str, Any]): The hyperparameter dictionary for calculation.
            cache (Optional[FeatureCache], optional): Shared cache instance used to
                fetch or store dependency series. Defaults to None.

        Returns:
            FeatureResult: Dataclass containing the computed data in the appropriate
                fields (data, levels, zones, heatmaps) based on output_schema.

        Raises:
            ValueError: If necessary market data columns are missing.
            FeatureError: If the function attempts an in-place mutation of the DataFrame.
        """
        pass
