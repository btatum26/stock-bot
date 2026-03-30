"""Base classes and data structures for the quantitative feature engineering system.

This module defines the core contracts, output structures, and global registry 
used to build and compute financial indicators or features consistently.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Type, TYPE_CHECKING
import pandas as pd

if TYPE_CHECKING:
    from .features import FeatureCache

# Global Registry
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

# --- Output Types ---
@dataclass
class FeatureResult:
    """Strict payload returned by a Feature's compute method.
    
    Attributes:
        visuals (List[FeatureOutput]): GUI rendering instructions.
        data (Dict[str, pd.Series]): The raw numerical series intended for the ML bridge.
    """
    data: Dict[str, pd.Series] = None 

# --- Base Feature Class ---
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

    @property
    def outputs(self) -> List[Optional[str]]:
        """Defines the raw data column suffixes produced by this feature.

        Returns:
            List[Optional[str]]: A list of string suffixes, or a list containing 
                None if the feature produces a single primary output.
        """
        return [None]

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
            FeatureResult: Dataclass containing visual instructions and raw data.

        Raises:
            ValueError: If necessary market data columns are missing.
            FeatureError: If the function attempts an in-place mutation of the DataFrame.
        """
        pass