import pytest
import pandas as pd
import numpy as np
from engine.core.features.base import Feature, FeatureResult, register_feature
from engine.core.features.features import compute_all_features
from engine.core.exceptions import FeatureError

@register_feature("MemoryViolator")
class MemoryViolator(Feature):
    @property
    def name(self): return "MemoryViolator"
    @property
    def description(self): return "Violates memory"
    @property
    def category(self): return "Test"

    def compute(self, df: pd.DataFrame, params: dict, cache) -> FeatureResult:
        # In-place assignment that triggers the memory violation check
        df["violator"] = df["close"] * 2
        return FeatureResult(data={"violator": df["violator"]})

def test_feature_memory_violation_catch():
    df = pd.DataFrame({"close": np.random.randn(100)})
    
    # Simulate a read-only environment
    df.values.setflags(write=False)
    
    feature_config = [
        {"id": "MemoryViolator", "params": {}}
    ]
    
    with pytest.raises(FeatureError) as excinfo:
        compute_all_features(df, feature_config)
    
    assert "attempted to mutate the input DataFrame in place" in str(excinfo.value)
    assert "MemoryViolator" in str(excinfo.value)

def test_feature_dependency_memory_violation_catch():
    @register_feature("DependencyViolator")
    class DependencyViolator(Feature):
        @property
        def name(self): return "DependencyViolator"
        @property
        def description(self): return "Violates memory via dependency"
        @property
        def category(self): return "Test"

        def compute(self, df: pd.DataFrame, params: dict, cache) -> FeatureResult:
            # Triggering dependency that violates memory
            cache.get_series("MemoryViolator", {}, df)
            return FeatureResult(data={"dummy": df["close"]}, visuals=[])

    df = pd.DataFrame({"close": np.random.randn(100)})
    df.values.setflags(write=False)
    
    feature_config = [
        {"id": "DependencyViolator", "params": {}}
    ]
    
    # Orchestrator catches the underlying FeatureError from get_series and re-raises
    with pytest.raises(FeatureError) as excinfo:
        compute_all_features(df, feature_config)
    
    assert "Feature computation failed for DependencyViolator" in str(excinfo.value)
