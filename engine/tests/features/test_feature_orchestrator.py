import pytest
import pandas as pd
import numpy as np
from engine.core.features.features import FeatureOrchestrator, FeatureCache
from engine.core.features.base import Feature, FeatureResult, register_feature, FEATURE_REGISTRY
from engine.core.exceptions import FeatureError

# Mock feature for testing
@register_feature("mock_feature")
class MockFeature(Feature):
    @property
    def name(self): return "Mock"
    @property
    def description(self): return "Mock"
    @property
    def category(self): return "Test"
    
    def compute(self, df, params, cache = None):
        window = params.get("window", 5)
        # Correct way: return a new series
        series = df['Close'].rolling(window=window).mean()
        return FeatureResult(data={"mock_feature": series})

@register_feature("bad_feature")
class BadFeature(Feature):
    @property
    def name(self): return "Bad"
    @property
    def description(self): return "Bad"
    @property
    def category(self): return "Test"
    
    def compute(self, df, params, cache = None):
        # INCORRECT: modifying df in place
        df['bad'] = df['Close'] * 2
        return FeatureResult(data={"bad": df['bad']})

@register_feature("dependent_feature")
class DependentFeature(Feature):
    @property
    def name(self): return "Dependent"
    @property
    def description(self): return "Dependent"
    @property
    def category(self): return "Test"
    
    def compute(self, df, params, cache = None):
        # Depends on mock_feature
        dep_series = cache.get_series("mock_feature", {"window": 5}, df)
        result = dep_series * 2
        return FeatureResult(data={"dependent_feature": result})

@pytest.fixture
def orchestrator():
    return FeatureOrchestrator()

@pytest.fixture
def sample_df():
    dates = pd.date_range('2023-01-01', periods=10)
    return pd.DataFrame({'Close': [float(x) for x in range(10)]}, index=dates)

def test_feature_orchestrator_compute(orchestrator, sample_df):
    config = [{"id": "mock_feature", "params": {"window": 3}}]
    df_out, l_max = orchestrator.compute_features(sample_df, config)
    
    assert "mock_feature" in df_out.columns
    assert df_out["mock_feature"].iloc[2] == 1.0 # (0+1+2)/3
    assert l_max == 3

def test_feature_blast_shield(orchestrator, sample_df):
    config = [{"id": "bad_feature", "params": {}}]
    with pytest.raises(FeatureError, match="attempted to mutate the input DataFrame in place"):
        orchestrator.compute_features(sample_df, config)

def test_feature_dependency(orchestrator, sample_df):
    config = [
        {"id": "mock_feature", "params": {"window": 5}},
        {"id": "dependent_feature", "params": {}}
    ]
    df_out, _ = orchestrator.compute_features(sample_df, config)
    
    assert "mock_feature" in df_out.columns
    assert "dependent_feature" in df_out.columns
    # dependent_feature = mock_feature * 2
    # mock_feature at index 4 is (0+1+2+3+4)/5 = 2.0
    assert df_out["dependent_feature"].iloc[4] == 4.0

def test_cache_key_generation():
    cache = FeatureCache()
    key1 = cache._generate_key("feat", {"a": 1, "b": 2})
    key2 = cache._generate_key("feat", {"b": 2, "a": 1})
    assert key1 == key2
    assert key1 == "feat_a1_b2"
