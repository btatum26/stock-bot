import pytest
import pandas as pd
from engine.core.features.features import FeatureCache, FeatureOrchestrator
from engine.core.features.base import Feature, FeatureResult, register_feature
from unittest.mock import MagicMock

def test_deterministic_key_generation():
    cache = FeatureCache()
    params1 = {"period": 14, "color": "red"}
    params2 = {"color": "red", "period": 14}
    
    # Internal method check
    key1 = cache._generate_key("EMA", params1)
    key2 = cache._generate_key("EMA", params2)
    
    assert key1 == key2
    assert "colorred" in key1
    assert "period14" in key1

@register_feature("MockFeature")
class MockFeature(Feature):
    compute_count = 0
    @property
    def name(self): return "MockFeature"
    @property
    def description(self): return "Mock"
    @property
    def category(self): return "Test"
    
    def compute(self, df, params, cache):
        MockFeature.compute_count += 1
        return FeatureResult(data={"mock": df["close"]})

def test_feature_cache_hit():
    cache = FeatureCache()
    df = pd.DataFrame({"close": [1, 2, 3]})
    MockFeature.compute_count = 0
    
    # First call - should compute
    res1 = cache.get_series("MockFeature", {"p": 1}, df)
    assert MockFeature.compute_count == 1
    
    # Second call with same params - should hit cache
    res2 = cache.get_series("MockFeature", {"p": 1}, df)
    assert MockFeature.compute_count == 1
    assert res1 is res2 # Should return the same object (the series)
    
    # Third call with different params - should compute again
    res3 = cache.get_series("MockFeature", {"p": 2}, df)
    assert MockFeature.compute_count == 2
