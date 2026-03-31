"""Tests for SignalValidator.validate_and_compress().

Covers: all three compression modes (clip, tanh, probability), unknown mode fallback,
list/numpy/Series inputs, Inf handling, NaN handling, index alignment, index mismatch
reindex, non-numeric string coercion, output name convention.
"""
import pytest
import pandas as pd
import numpy as np

from engine.core.backtester import SignalValidator
from engine.core.exceptions import StrategyError


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def idx():
    return pd.date_range("2023-01-01", periods=5, freq="B")


# ---------------------------------------------------------------------------
# Input type coercion
# ---------------------------------------------------------------------------

class TestInputCoercion:
    def test_pandas_series_accepted(self, idx):
        raw = pd.Series([0.5, -0.5, 1.0, -1.0, 0.0], index=idx)
        out = SignalValidator.validate_and_compress(raw, idx)
        assert isinstance(out, pd.Series)

    def test_list_accepted(self, idx):
        raw = [0.5, -0.5, 1.0, -1.0, 0.0]
        out = SignalValidator.validate_and_compress(raw, idx)
        assert isinstance(out, pd.Series)
        assert list(out.index) == list(idx)

    def test_numpy_array_accepted(self, idx):
        raw = np.array([0.5, -0.5, 1.0, -1.0, 0.0])
        out = SignalValidator.validate_and_compress(raw, idx)
        assert isinstance(out, pd.Series)

    def test_non_numeric_strings_become_zero(self, idx):
        raw = ["a", "b", "c", "d", "e"]
        out = SignalValidator.validate_and_compress(raw, idx)
        assert (out == 0.0).all()

    def test_output_named_conviction_signal(self, idx):
        raw = pd.Series([0.0] * 5, index=idx)
        out = SignalValidator.validate_and_compress(raw, idx)
        assert out.name == "conviction_signal"


# ---------------------------------------------------------------------------
# Inf and NaN handling
# ---------------------------------------------------------------------------

class TestInvalidValues:
    def test_inf_replaced_with_zero(self, idx):
        raw = pd.Series([np.inf, -np.inf, 0.5, -0.5, 1.0], index=idx)
        out = SignalValidator.validate_and_compress(raw, idx)
        assert out.iloc[0] == 0.0
        assert out.iloc[1] == 0.0

    def test_nan_replaced_with_zero(self, idx):
        raw = pd.Series([np.nan, 0.5, np.nan, -0.5, 1.0], index=idx)
        out = SignalValidator.validate_and_compress(raw, idx)
        assert out.iloc[0] == 0.0
        assert out.iloc[2] == 0.0

    def test_mixed_nan_inf(self, idx):
        raw = pd.Series([np.nan, np.inf, -np.inf, 0.3, np.nan], index=idx)
        out = SignalValidator.validate_and_compress(raw, idx)
        assert np.isfinite(out).all()


# ---------------------------------------------------------------------------
# Index alignment
# ---------------------------------------------------------------------------

class TestIndexAlignment:
    def test_list_gets_target_index(self, idx):
        raw = [1.0, -1.0, 0.5, -0.5, 0.0]
        out = SignalValidator.validate_and_compress(raw, idx)
        assert out.index.equals(idx)

    def test_series_with_wrong_int_index_gets_reindexed(self, idx):
        raw = pd.Series([0.5, -0.5, 1.0, -1.0, 0.0])  # default RangeIndex
        out = SignalValidator.validate_and_compress(raw, idx)
        assert out.index.equals(idx)

    def test_series_shorter_than_target_reindexed_with_zeros(self, idx):
        short_idx = pd.date_range("2023-01-01", periods=3, freq="B")
        raw = pd.Series([1.0, -1.0, 0.5], index=short_idx)
        out = SignalValidator.validate_and_compress(raw, idx)
        # Length should match target; missing positions filled with 0.0
        assert len(out) == len(idx)
        assert (out.isna() == False).all()


# ---------------------------------------------------------------------------
# Compression modes
# ---------------------------------------------------------------------------

class TestClipMode:
    def test_values_above_one_clipped(self, idx):
        raw = pd.Series([5.0, 2.0, 1.5, 0.5, -0.5], index=idx)
        out = SignalValidator.validate_and_compress(raw, idx, compression_mode="clip")
        assert out.max() <= 1.0

    def test_values_below_neg_one_clipped(self, idx):
        raw = pd.Series([-5.0, -2.0, -1.5, -0.5, 0.5], index=idx)
        out = SignalValidator.validate_and_compress(raw, idx, compression_mode="clip")
        assert out.min() >= -1.0

    def test_in_range_values_unchanged(self, idx):
        raw = pd.Series([0.3, -0.3, 0.7, -0.7, 0.0], index=idx)
        out = SignalValidator.validate_and_compress(raw, idx, compression_mode="clip")
        pd.testing.assert_series_equal(
            out.reset_index(drop=True),
            raw.reset_index(drop=True).rename("conviction_signal"),
        )

    def test_default_mode_is_clip(self, idx):
        raw = pd.Series([3.0, -3.0, 0.5, -0.5, 0.0], index=idx)
        out_explicit = SignalValidator.validate_and_compress(raw, idx, compression_mode="clip")
        out_default = SignalValidator.validate_and_compress(raw, idx)
        pd.testing.assert_series_equal(out_explicit, out_default)


class TestTanhMode:
    def test_output_bounded(self, idx):
        raw = pd.Series([100.0, -100.0, 1.0, -1.0, 0.0], index=idx)
        out = SignalValidator.validate_and_compress(raw, idx, compression_mode="tanh")
        assert out.max() <= 1.0
        assert out.min() >= -1.0

    def test_zero_maps_to_zero(self, idx):
        raw = pd.Series([0.0] * 5, index=idx)
        out = SignalValidator.validate_and_compress(raw, idx, compression_mode="tanh")
        assert (out == 0.0).all()

    def test_positive_stays_positive(self, idx):
        raw = pd.Series([0.5, 1.0, 2.0, 0.1, 0.3], index=idx)
        out = SignalValidator.validate_and_compress(raw, idx, compression_mode="tanh")
        assert (out > 0).all()

    def test_values_equal_tanh_of_input(self, idx):
        raw = pd.Series([0.5, -0.5, 1.0, -1.0, 0.0], index=idx)
        out = SignalValidator.validate_and_compress(raw, idx, compression_mode="tanh")
        expected = np.tanh(raw.values)
        np.testing.assert_allclose(out.values, expected, atol=1e-9)


class TestProbabilityMode:
    def test_half_maps_to_zero(self, idx):
        raw = pd.Series([0.5] * 5, index=idx)
        out = SignalValidator.validate_and_compress(raw, idx, compression_mode="probability")
        np.testing.assert_allclose(out.values, 0.0, atol=1e-9)

    def test_one_maps_to_one(self, idx):
        raw = pd.Series([1.0] * 5, index=idx)
        out = SignalValidator.validate_and_compress(raw, idx, compression_mode="probability")
        np.testing.assert_allclose(out.values, 1.0, atol=1e-9)

    def test_zero_maps_to_neg_one(self, idx):
        raw = pd.Series([0.0] * 5, index=idx)
        out = SignalValidator.validate_and_compress(raw, idx, compression_mode="probability")
        np.testing.assert_allclose(out.values, -1.0, atol=1e-9)

    def test_linear_mapping(self, idx):
        raw = pd.Series([0.0, 0.25, 0.5, 0.75, 1.0], index=idx)
        out = SignalValidator.validate_and_compress(raw, idx, compression_mode="probability")
        expected = (raw.values * 2) - 1.0
        np.testing.assert_allclose(out.values, expected, atol=1e-9)


class TestUnknownMode:
    def test_unknown_mode_falls_back_to_clip(self, idx):
        raw = pd.Series([5.0, -5.0, 0.5, -0.5, 0.0], index=idx)
        out_unknown = SignalValidator.validate_and_compress(raw, idx, compression_mode="bogus_mode")
        out_clip = SignalValidator.validate_and_compress(raw, idx, compression_mode="clip")
        pd.testing.assert_series_equal(out_unknown, out_clip)
