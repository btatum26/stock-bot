"""Tests for the feature registry and load_features().

Covers: registry is populated after load_features(), expected engine features are
registered, all registered features conform to the Feature interface (name,
description, category, parameters, output_schema, compute), and generate_column_name
produces deterministic, collision-resistant identifiers.
"""
import pytest
import pandas as pd
import numpy as np

from engine.core.features.features import load_features, FeatureCache
from engine.core.features.base import (
    FEATURE_REGISTRY,
    Feature,
    FeatureResult,
    OutputSchema,
    OutputType,
    Pane,
)


# ---------------------------------------------------------------------------
# Registry population
# ---------------------------------------------------------------------------

class TestRegistryPopulation:
    def test_registry_not_empty_after_load(self):
        load_features()
        assert len(FEATURE_REGISTRY) > 0

    def test_all_values_are_feature_subclasses(self):
        load_features()
        for feat_id, feat_cls in FEATURE_REGISTRY.items():
            assert issubclass(feat_cls, Feature), (
                f"{feat_id} is not a Feature subclass"
            )

    @pytest.mark.parametrize("feature_id", [
        "RSI",
        "BollingerBands",
        "SupportResistance",
        "SMA",
        "EMA",
        "MACD",
        "ATR",
        "Stochastic",
        "OBV",
        "VWAP",
        "CCI",
        "Fibonacci",
        "Ichimoku",
        "Supertrend",
        "LinReg",
        "CandlePatterns",
        "KDE",
        "VolumeProfile",
        "Volume",
    ])
    def test_known_feature_registered(self, feature_id):
        load_features()
        assert feature_id in FEATURE_REGISTRY, (
            f"Expected feature '{feature_id}' to be in FEATURE_REGISTRY"
        )


# ---------------------------------------------------------------------------
# Feature interface compliance
# ---------------------------------------------------------------------------

class TestFeatureInterface:
    """Every feature in the registry must satisfy the base Feature contract."""

    @pytest.fixture(scope="class", autouse=True)
    def ensure_loaded(self):
        load_features()

    def test_all_features_have_string_name(self):
        for feat_id, feat_cls in FEATURE_REGISTRY.items():
            instance = feat_cls()
            assert isinstance(instance.name, str) and instance.name, (
                f"{feat_id}.name must be a non-empty string"
            )

    def test_all_features_have_string_description(self):
        for feat_id, feat_cls in FEATURE_REGISTRY.items():
            instance = feat_cls()
            assert isinstance(instance.description, str), (
                f"{feat_id}.description must be a string"
            )

    def test_all_features_have_string_category(self):
        for feat_id, feat_cls in FEATURE_REGISTRY.items():
            instance = feat_cls()
            assert isinstance(instance.category, str) and instance.category, (
                f"{feat_id}.category must be a non-empty string"
            )

    def test_all_features_have_dict_parameters(self):
        for feat_id, feat_cls in FEATURE_REGISTRY.items():
            instance = feat_cls()
            assert isinstance(instance.parameters, dict), (
                f"{feat_id}.parameters must return a dict"
            )

    def test_all_features_have_output_schema_list(self):
        for feat_id, feat_cls in FEATURE_REGISTRY.items():
            instance = feat_cls()
            schema = instance.output_schema
            assert isinstance(schema, list), (
                f"{feat_id}.output_schema must return a list"
            )
            assert len(schema) > 0, (
                f"{feat_id}.output_schema must not be empty"
            )

    def test_all_output_schemas_are_valid(self):
        for feat_id, feat_cls in FEATURE_REGISTRY.items():
            instance = feat_cls()
            for entry in instance.output_schema:
                assert isinstance(entry, OutputSchema), (
                    f"{feat_id}: output_schema entries must be OutputSchema instances"
                )
                assert isinstance(entry.output_type, OutputType), (
                    f"{feat_id}: OutputSchema.output_type must be an OutputType"
                )
                assert isinstance(entry.pane, Pane), (
                    f"{feat_id}: OutputSchema.pane must be a Pane"
                )


# ---------------------------------------------------------------------------
# Column name generation
# ---------------------------------------------------------------------------

class TestColumnNameGeneration:
    @pytest.fixture(autouse=True)
    def load(self):
        load_features()

    def test_deterministic_across_calls(self):
        feat = FEATURE_REGISTRY["RSI"]()
        params = {"period": 14}
        name1 = feat.generate_column_name("RSI", params, "value")
        name2 = feat.generate_column_name("RSI", params, "value")
        assert name1 == name2

    def test_different_params_produce_different_names(self):
        feat = FEATURE_REGISTRY["RSI"]()
        name14 = feat.generate_column_name("RSI", {"period": 14}, "value")
        name21 = feat.generate_column_name("RSI", {"period": 21}, "value")
        assert name14 != name21

    def test_different_suffixes_produce_different_names(self):
        feat = FEATURE_REGISTRY["BollingerBands"]()
        params = {"period": 20, "std_dev": 2.0}
        upper = feat.generate_column_name("BollingerBands", params, "upper")
        lower = feat.generate_column_name("BollingerBands", params, "lower")
        assert upper != lower

    def test_column_name_contains_feature_id(self):
        feat = FEATURE_REGISTRY["RSI"]()
        name = feat.generate_column_name("RSI", {"period": 14}, "value")
        assert "RSI" in name or "rsi" in name.lower()


# ---------------------------------------------------------------------------
# Smoke-test compute() for key features
# ---------------------------------------------------------------------------

@pytest.fixture
def ohlcv_df():
    """50-bar OHLCV DataFrame — enough for most indicator warmup periods."""
    n = 50
    rng = np.random.default_rng(0)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({
        "Open":   close + rng.normal(0, 0.3, n),
        "High":   close + abs(rng.normal(0, 0.5, n)),
        "Low":    close - abs(rng.normal(0, 0.5, n)),
        "Close":  close,
        "Volume": rng.integers(100_000, 1_000_000, n).astype(float),
    }, index=dates)


@pytest.mark.parametrize("feature_id,params", [
    ("RSI",           {"period": 14}),
    ("BollingerBands",{"period": 20, "std_dev": 2.0}),
    ("SMA",           {"period": 10}),
    ("EMA",           {"period": 10}),
    ("ATR",           {"period": 14}),
    ("MACD",          {"fast": 12, "slow": 26, "signal": 9}),
    ("Stochastic",    {"period": 14}),
    ("OBV",           {}),
    ("CCI",           {"period": 20}),
    ("Volume",        {}),
])
def test_feature_compute_returns_feature_result(feature_id, params, ohlcv_df):
    """Smoke-test: feature compute() returns FeatureResult with data."""
    load_features()
    feat_cls = FEATURE_REGISTRY[feature_id]
    instance = feat_cls()
    cache = FeatureCache()
    result = instance.compute(ohlcv_df, params, cache)

    assert isinstance(result, FeatureResult)
    assert result.data is not None
    assert len(result.data) > 0

    for col, series in result.data.items():
        assert isinstance(series, pd.Series), (
            f"{feature_id}: data['{col}'] must be a pd.Series"
        )
        assert len(series) == len(ohlcv_df), (
            f"{feature_id}: data['{col}'] length mismatch"
        )


@pytest.mark.parametrize("feature_id,params", [
    ("SupportResistance", {"method": "Bill Williams", "window": 3}),
    ("Fibonacci",         {}),
])
def test_level_features_return_levels(feature_id, params, ohlcv_df):
    """Level-type features should populate FeatureResult.levels."""
    load_features()
    feat_cls = FEATURE_REGISTRY[feature_id]
    instance = feat_cls()
    cache = FeatureCache()
    result = instance.compute(ohlcv_df, params, cache)

    assert isinstance(result, FeatureResult)
    # levels may be empty if no pivots found in short data, but must be a list
    assert result.levels is None or isinstance(result.levels, list)
