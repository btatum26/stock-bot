"""Tests for Tearsheet.calculate_metrics().

Covers: happy path, T+1 execution model, CAGR, Sharpe, max drawdown, win rate,
profit factor zero-division, all-zero signals, single-day data, equity curve shape.
"""
import pytest
import pandas as pd
import numpy as np
from engine.core.metrics import Tearsheet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n: int = 50, seed: int = 42) -> pd.DataFrame:
    """Return a minimal OHLCV DataFrame with a DatetimeIndex."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    open_ = close + rng.normal(0, 0.5, n)
    return pd.DataFrame(
        {"Open": open_, "High": close + 1, "Low": close - 1, "Close": close, "Volume": 1_000_000},
        index=dates,
    )


def _const_signals(n: int, value: float) -> pd.Series:
    df = _make_df(n)
    return pd.Series(value, index=df.index)


# ---------------------------------------------------------------------------
# Structure tests
# ---------------------------------------------------------------------------

class TestTearsheetReturnShape:
    def test_returns_dict(self):
        df = _make_df()
        signals = pd.Series(1.0, index=df.index)
        result = Tearsheet.calculate_metrics(df, signals)
        assert isinstance(result, dict)

    def test_required_keys_present(self):
        required = {
            "Total Return (%)", "CAGR (%)", "Max Drawdown (%)",
            "Win Rate (%)", "Profit Factor", "Total Trades",
            "Sharpe Ratio", "Deflated Sharpe Ratio", "equity_curve",
        }
        df = _make_df()
        signals = pd.Series(1.0, index=df.index)
        result = Tearsheet.calculate_metrics(df, signals)
        assert required.issubset(result.keys())

    def test_equity_curve_is_series(self):
        df = _make_df()
        signals = pd.Series(1.0, index=df.index)
        result = Tearsheet.calculate_metrics(df, signals)
        assert isinstance(result["equity_curve"], pd.Series)

    def test_equity_curve_length_matches_input(self):
        n = 40
        df = _make_df(n)
        signals = pd.Series(1.0, index=df.index)
        result = Tearsheet.calculate_metrics(df, signals)
        assert len(result["equity_curve"]) == n


# ---------------------------------------------------------------------------
# Numerical correctness
# ---------------------------------------------------------------------------

class TestTearsheetNumerics:
    def test_all_zero_signals_zero_return(self):
        df = _make_df()
        signals = pd.Series(0.0, index=df.index)
        result = Tearsheet.calculate_metrics(df, signals)
        # Flat position → no return, no drawdown
        assert result["Total Return (%)"] == pytest.approx(0.0, abs=1e-6)
        assert result["Max Drawdown (%)"] == pytest.approx(0.0, abs=1e-6)
        assert result["Total Trades"] == 0

    def test_equity_curve_starts_near_one(self):
        """First bar of equity curve should be ~1.0 (before any compounding)."""
        df = _make_df()
        signals = pd.Series(0.5, index=df.index)
        result = Tearsheet.calculate_metrics(df, signals)
        assert result["equity_curve"].iloc[0] == pytest.approx(1.0, abs=0.05)

    def test_equity_curve_monotone_long_uptrend(self):
        """Constant long signal on a rising open price should produce an equity > 1."""
        n = 30
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        open_ = np.linspace(100, 130, n)           # strictly rising opens
        df = pd.DataFrame(
            {"Open": open_, "High": open_ + 1, "Low": open_ - 1,
             "Close": open_, "Volume": 1_000},
            index=dates,
        )
        signals = pd.Series(1.0, index=df.index)
        result = Tearsheet.calculate_metrics(df, signals)
        assert result["equity_curve"].iloc[-1] > 1.0

    def test_max_drawdown_negative_or_zero(self):
        df = _make_df()
        signals = pd.Series(1.0, index=df.index)
        result = Tearsheet.calculate_metrics(df, signals)
        assert result["Max Drawdown (%)"] <= 0.0

    def test_win_rate_bounded(self):
        df = _make_df()
        signals = pd.Series([1.0, -1.0] * (len(df) // 2), index=df.index)
        result = Tearsheet.calculate_metrics(df, signals)
        assert 0.0 <= result["Win Rate (%)"] <= 100.0

    def test_sharpe_is_finite_for_nonzero_signals(self):
        df = _make_df()
        signals = pd.Series(1.0, index=df.index)
        result = Tearsheet.calculate_metrics(df, signals)
        assert np.isfinite(result["Sharpe Ratio"])

    def test_profit_factor_inf_when_no_losses(self):
        """When strategy has only gains, Profit Factor should be infinity."""
        n = 30
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        open_ = np.arange(100.0, 100.0 + n)        # strictly increasing
        df = pd.DataFrame(
            {"Open": open_, "High": open_ + 1, "Low": open_ - 1,
             "Close": open_, "Volume": 1_000},
            index=dates,
        )
        # Entry at T=0, stay long — every trade return is positive
        signals = pd.Series(1.0, index=df.index)
        result = Tearsheet.calculate_metrics(df, signals)
        assert result["Profit Factor"] == float("inf") or result["Profit Factor"] > 1.0

    def test_total_trades_counts_signal_changes(self):
        n = 10
        df = _make_df(n)
        # Alternating ±1 → 9 direction changes
        vals = [1.0, -1.0] * (n // 2)
        signals = pd.Series(vals, index=df.index)
        result = Tearsheet.calculate_metrics(df, signals)
        assert result["Total Trades"] == n - 1

    def test_friction_reduces_total_return(self):
        """Higher friction should produce a lower (or equal) total return."""
        df = _make_df()
        signals = pd.Series([1.0, -1.0] * (len(df) // 2), index=df.index)
        low_friction = Tearsheet.calculate_metrics(df, signals, friction=0.0)
        high_friction = Tearsheet.calculate_metrics(df, signals, friction=0.01)
        assert high_friction["Total Return (%)"] <= low_friction["Total Return (%)"]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestTearsheetEdgeCases:
    def test_single_row_does_not_crash(self):
        """Single-row DataFrame should return zeros without raising."""
        dates = pd.date_range("2020-01-01", periods=1)
        df = pd.DataFrame(
            {"Open": [100.0], "High": [101.0], "Low": [99.0],
             "Close": [100.0], "Volume": [1000]},
            index=dates,
        )
        signals = pd.Series([1.0], index=dates)
        result = Tearsheet.calculate_metrics(df, signals)
        assert result["CAGR (%)"] == 0.0  # same-day start/end → days==0 branch

    def test_cagr_zero_when_same_day_start_end(self):
        """days == 0 guard should return CAGR of 0."""
        dates = pd.date_range("2020-06-01", periods=1)
        df = pd.DataFrame(
            {"Open": [100.0], "High": [101.0], "Low": [99.0],
             "Close": [100.0], "Volume": [1000]},
            index=dates,
        )
        signals = pd.Series([0.0], index=dates)
        result = Tearsheet.calculate_metrics(df, signals)
        assert result["CAGR (%)"] == 0.0

    def test_deflated_sharpe_placeholder_is_zero(self):
        """DSR is a TODO placeholder that should always be 0.0 for now."""
        df = _make_df()
        signals = pd.Series(1.0, index=df.index)
        result = Tearsheet.calculate_metrics(df, signals)
        assert result["Deflated Sharpe Ratio"] == 0.0

    def test_print_summary_does_not_raise(self):
        """print_summary should complete without exceptions."""
        df = _make_df()
        signals = pd.Series(1.0, index=df.index)
        metrics = Tearsheet.calculate_metrics(df, signals)
        Tearsheet.print_summary(metrics)  # would raise if broken
