"""Feature execution and orchestration framework.

This module manages the dynamic loading, dependency caching, and safe 
execution of computational features across a primary dataset.
"""

import os
import importlib
import pkgutil
import pandas as pd
from typing import List, Dict, Any, Optional
from .base import FEATURE_REGISTRY, FeatureResult, Feature
from ..logger import logger
from ..exceptions import FeatureError, ValidationError

def load_features():
    """Dynamically loads and registers all feature modules.
    
    This function traverses the features directory, importing each module 
    to trigger the `@register_feature` decorators, thereby populating the 
    global feature registry. Base and utility modules are skipped to avoid 
    circular dependencies.
    """
    base_dir = os.path.dirname(__file__)
    for loader, module_name, is_pkg in pkgutil.walk_packages([base_dir], prefix="engine.core.features."):
        if not is_pkg:
            if module_name.endswith(".base") or module_name.endswith(".features"):
                continue
            try:
                importlib.import_module(module_name)
            except Exception as e:
                logger.error(f"Failed to load feature module {module_name}: {e}")

load_features()

class FeatureCache:
    """Manages in-memory caching of computed feature series.
    
    This system prevents redundant computation of shared dependencies (like moving 
    averages) across multiple distinct features during an orchestration pass.
    """
    
    def __init__(self):
        """Initializes the empty dictionary used for memory storage."""
        self._memory: Dict[str, pd.Series] = {}

    def _generate_key(self, feature_id: str, params: Dict[str, Any]) -> str:
        """Generates a unique cache key based on feature ID and parameters.

        Args:
            feature_id (str): The unique identifier of the feature.
            params (Dict[str, Any]): The parameters used for computation.

        Returns:
            str: A deterministic string key.
        """
        if not params:
            return feature_id
        param_str = "_".join([f"{k}{v}" for k, v in sorted(params.items())])
        return f"{feature_id}_{param_str}"

    def get_series(self, feature_id: str, params: Dict[str, Any], df: pd.DataFrame) -> pd.Series:
        """Retrieves a feature series from cache or computes it if missing.

        Args:
            feature_id (str): The string identifier of the requested feature.
            params (Dict[str, Any]): The required parameters for computation.
            df (pd.DataFrame): The base market dataset.

        Returns:
            pd.Series: The fully computed feature data series.

        Raises:
            FeatureError: If the feature is not found, returns no data, or violates 
                memory safety by altering the base DataFrame.
        """
        key = self._generate_key(feature_id, params)
        
        if key in self._memory:
            return self._memory[key]
            
        if feature_id not in FEATURE_REGISTRY:
            raise FeatureError(f"Feature '{feature_id}' not found in registry.")
            
        feature_cls = FEATURE_REGISTRY[feature_id]
        feature_instance = feature_cls()
        
        initial_col_count = len(df.columns)
        try:
            result: FeatureResult = feature_instance.compute(df, params, self)
        except ValueError as e:
            if "read-only" in str(e).lower() or "not writable" in str(e).lower():
                self._raise_memory_violation(feature_id, is_dependency=True)
            raise e
        except Exception as e:
            logger.error(f"Error computing feature {feature_id}: {e}")
            raise FeatureError(f"Feature computation failed for {feature_id}")

        if len(df.columns) != initial_col_count:
            self._raise_memory_violation(feature_id, is_dependency=True)

        if not result.data:
            raise FeatureError(f"Feature '{feature_id}' returned no data.")
            
        primary_series = None
        for col_name, series in result.data.items():
            self._memory[col_name] = series
            if primary_series is None:
                primary_series = series 
                
        self._memory[key] = primary_series
        return primary_series

    def set_series(self, key: str, series: pd.Series):
        """Manually injects a computed series into the memory cache.

        Args:
            key (str): The unique retrieval key.
            series (pd.Series): The computed data to store.
        """
        self._memory[key] = series

    def _raise_memory_violation(self, feature_id: str, is_dependency: bool = False):
        """Raises a standardized error when in-place DataFrame mutations are detected.

        Args:
            feature_id (str): The identifier of the violating feature.
            is_dependency (bool, optional): Indicates if the violation occurred while 
                computing a cached dependency. Defaults to False.

        Raises:
            FeatureError: The formulated error message regarding memory modification.
        """
        dep_str = "dependency " if is_dependency else ""
        error_msg = (
            f"Feature {dep_str}'{feature_id}' attempted to mutate the input DataFrame in place. "
            f"Features must return new Series objects instead of assigning new columns to the input df."
        )
        logger.error(error_msg)
        raise FeatureError(error_msg)

class FeatureOrchestrator:
    """Handles the batch computation and safety validation of multiple features."""

    def validate_config(self, feature_config: List[Dict[str, Any]]):
        """Validates the structure and validity of the requested feature configuration.

        Args:
            feature_config (List[Dict[str, Any]]): The list of feature request configurations.

        Raises:
            ValidationError: If a configuration lacks an ID, or if the ID does not 
                exist in the active registry.
        """
        for config in feature_config:
            feature_id = config.get("id")
            if not feature_id:
                raise ValidationError("Feature configuration missing 'id' field.")
            if feature_id not in FEATURE_REGISTRY:
                raise ValidationError(f"Feature '{feature_id}' not found in registry.")

    def compute_features(self, df: pd.DataFrame, feature_config: List[Dict[str, Any]]) -> tuple[pd.DataFrame, int]:
        """Executes a batch computation of multiple features sequentially.

        This orchestrator iterates through a configuration of requested features, 
        instantiates them from the global registry, and manages a shared cache to 
        optimize dependencies. It strictly enforces memory safety by ensuring no 
        feature mutates the original dataset.

        Args:
            df (pd.DataFrame): The base OHLCV dataset. 
            feature_config (List[Dict[str, Any]]): A list of feature request payload 
                dictionaries. Example: `[{"id": "RSI", "params": {"window": 14}}]`.

        Returns:
            tuple:
                - pd.DataFrame: A newly concatenated DataFrame containing the original 
                  OHLCV data alongside all newly computed feature columns.
                - int: The maximum lookback window discovered across all feature 
                  parameters, required for safe data truncation downstream.

        Raises:
            ValidationError: If configuration validation fails.
            FeatureError: If a child feature attempts an in-place mutation, or 
                if general computation fails.
        """
        self.validate_config(feature_config)

        computed_features = {}
        l_max = 0
        lookback_keys = ["window", "period", "slow", "fast", "lookback"]

        cache = FeatureCache()

        for config in feature_config:
            feature_id = config.get("id")
            params = config.get("params", {})

            for k, v in params.items():
                if k.lower() in lookback_keys and isinstance(v, (int, float)):
                    l_max = max(l_max, int(v))

            feature_cls = FEATURE_REGISTRY[feature_id]
            feature_instance = feature_cls()

            initial_col_count = len(df.columns)
            try:
                result: FeatureResult = feature_instance.compute(df, params, cache)
            except ValueError as e:
                if "read-only" in str(e).lower() or "not writable" in str(e).lower():
                    self._raise_memory_violation(feature_id)
                raise e
            except Exception as e:
                logger.error(f"Orchestrator failed to compute feature {feature_id}: {e}")
                raise FeatureError(f"Feature computation failed for {feature_id}")

            if len(df.columns) != initial_col_count:
                self._raise_memory_violation(feature_id)

            if result.data:
                for col_name, series in result.data.items():
                    if col_name not in computed_features:
                        computed_features[col_name] = series
                        cache.set_series(col_name, series)

        if computed_features:
            new_features_df = pd.DataFrame(computed_features)
            df = pd.concat([df, new_features_df], axis=1)
            df = df.loc[:, ~df.columns.duplicated()]
            
        return df, l_max

    def _raise_memory_violation(self, feature_id: str):
        """Raises a standardized error when in-place DataFrame mutations are detected.

        Args:
            feature_id (str): The identifier of the violating feature.

        Raises:
            FeatureError: The formulated error message regarding memory modification.
        """
        error_msg = (
            f"Feature '{feature_id}' attempted to mutate the input DataFrame in place. "
            f"Features must return new Series objects instead of assigning new columns to the input df."
        )
        logger.error(error_msg)
        raise FeatureError(error_msg)

orchestrator = FeatureOrchestrator()

def compute_all_features(df: pd.DataFrame, feature_config: List[Dict[str, Any]]):
    """Utility function wrapper for high-level feature batch computation.

    Args:
        df (pd.DataFrame): The base market dataset.
        feature_config (List[Dict[str, Any]]): The list of requested features.

    Returns:
        tuple: See FeatureOrchestrator.compute_features for exact return types.
    """
    return orchestrator.compute_features(df, feature_config)