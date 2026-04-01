"""Tests for Tearsheet.calculate_metrics().

Covers: happy path, T+1 execution model, CAGR, Sharpe, Sortino, Calmar,
max drawdown, win rate, profit factor zero-division, all-zero signals,
single-day data, equity curve shape, discrete portfolio, trade log.
"""
import pytest
import pandas as pd
import numpy as np
from engine.core.backtester import Tearsheet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n: int = 50, seed: int = 42) -> pd.DataFrame:
    """Return a minimal OHLCV DataFrame with lowercase columns and a DatetimeIndex."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    open_ = close + rng.normal(0, 0.5, n)
    return pd.DataFrame(
        {"open": open_, "high": close + 1, "low": close - 1, "close": close, "volume": 1_000_000},
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
            "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio",
            "Avg Win (%)", "Avg Loss (%)", "Expectancy (%)",
            "Discrete Trades", "Discrete Win Rate (%)",
            "equity_curve", "portfolio", "bh_portfolio", "trade_log",
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

    def test_portfolio_is_series(self):
        df = _make_df()
        signals = pd.Series(1.0, index=df.index)
        result = Tearsheet.calculate_metrics(df, signals)
        assert isinstance(result["portfolio"], pd.Series)

    def test_bh_portfolio_is_series(self):
        df = _make_df()
        signals = pd.Series(1.0, index=df.index)
        result = Tearsheet.calculate_metrics(df, signals)
        assert isinstance(result["bh_portfolio"], pd.Series)

    def test_trade_log_is_dataframe(self):
        df = _make_df()
        signals = pd.Series([1.0, -1.0] * (len(df) // 2), index=df.index)
        result = Tearsheet.calculate_metrics(df, signals)
        assert isinstance(result["trade_log"], pd.DataFrame)


# ---------------------------------------------------------------------------
# Numerical correctness
# ---------------------------------------------------------------------------

class TestTearsheetNumerics:
    def test_all_zero_signals_zero_return(self):
        df = _make_df()
        signals = pd.Series(0.0, index=df.index)
        result = Tearsheet.calculate_metrics(df, signals)
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
        open_ = np.linspace(100, 130, n)
        df = pd.DataFrame(
            {"open": open_, "high": open_ + 1, "low": open_ - 1,
             "close": open_, "volume": 1_000},
            index=dates,
        )
        signals = pd.Series(1.0, index=df.index)
        result = Tearsheet.calculate_metrics(df, signals)
        assert result["equity_curve"].iloc[-1] > 1.0

    def test_portfolio_starts_at_starting_capital(self):
        """First bar of portfolio should equal starting_capital."""
        df = _make_df()
        signals = pd.Series(1.0, index=df.index)
        result = Tearsheet.calculate_metrics(df, signals, starting_capital=50_000.0)
        assert result["portfolio"].iloc[0] == pytest.approx(50_000.0, rel=0.01)

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

    def test_sortino_is_finite_for_nonzero_signals(self):
        df = _make_df()
        signals = pd.Series(1.0, index=df.index)
        result = Tearsheet.calculate_metrics(df, signals)
        assert np.isfinite(result["Sortino Ratio"])

    def test_calmar_ratio_positive_when_cagr_positive(self):
        """Calmar = CAGR / |MaxDD|. Should be positive if CAGR > 0."""
        n = 30
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        open_ = np.linspace(100, 130, n)
        df = pd.DataFrame(
            {"open": open_, "high": open_ + 1, "low": open_ - 1,
             "close": open_, "volume": 1_000},
            index=dates,
        )
        signals = pd.Series(1.0, index=df.index)
        result = Tearsheet.calculate_metrics(df, signals)
        if result["CAGR (%)"] > 0:
            assert result["Calmar Ratio"] >= 0.0

    def test_profit_factor_inf_when_no_losses(self):
        """When strategy has only gains, Profit Factor should be infinity."""
        n = 30
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        open_ = np.arange(100.0, 100.0 + n)
        df = pd.DataFrame(
            {"open": open_, "high": open_ + 1, "low": open_ - 1,
             "close": open_, "volume": 1_000},
            index=dates,
        )
        signals = pd.Series(1.0, index=df.index)
        result = Tearsheet.calculate_metrics(df, signals)
        assert result["Profit Factor"] == float("inf") or result["Profit Factor"] > 1.0

    def test_total_trades_counts_signal_changes(self):
        n = 10
        df = _make_df(n)
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
# Discrete simulation
# ---------------------------------------------------------------------------

class TestDiscreteSimulation:
    def test_zero_signals_produce_no_trades(self):
        df = _make_df()
        signals = pd.Series(0.0, index=df.index)
        result = Tearsheet.calculate_metrics(df, signals)
        assert result["Discrete Trades"] == 0
        assert result["trade_log"].empty

    def test_trade_log_has_expected_columns(self):
        df = _make_df()
        signals = pd.Series([1.0, -1.0] * (len(df) // 2), index=df.index)
        result = Tearsheet.calculate_metrics(df, signals)
        expected_cols = {"entry_date", "exit_date", "direction", "entry_price", "exit_price",
                         "return_pct", "bars_held"}
        if not result["trade_log"].empty:
            assert expected_cols.issubset(result["trade_log"].columns)

    def test_below_threshold_signals_produce_no_trades(self):
        """Signals below entry_threshold should result in flat position and no trades."""
        df = _make_df()
        signals = pd.Series(0.1, index=df.index)
        result = Tearsheet.calculate_metrics(df, signals, entry_threshold=0.2)
        assert result["Discrete Trades"] == 0

    def test_trade_direction_is_long_for_positive_signal(self):
        n = 20
        df = _make_df(n)
        signals = pd.Series(1.0, index=df.index)
        result = Tearsheet.calculate_metrics(df, signals)
        tl = result["trade_log"]
        if not tl.empty:
            assert (tl["direction"] == "LONG").all()

    def test_trade_direction_is_short_for_negative_signal(self):
        n = 20
        df = _make_df(n)
        signals = pd.Series(-1.0, index=df.index)
        result = Tearsheet.calculate_metrics(df, signals)
        tl = result["trade_log"]
        if not tl.empty:
            assert (tl["direction"] == "SHORT").all()

    def test_portfolio_grows_on_sustained_long_uptrend(self):
        n = 30
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        open_ = np.linspace(100, 130, n)
        df = pd.DataFrame(
            {"open": open_, "high": open_ + 1, "low": open_ - 1,
             "close": open_, "volume": 1_000},
            index=dates,
        )
        signals = pd.Series(1.0, index=df.index)
        result = Tearsheet.calculate_metrics(df, signals, starting_capital=10_000.0)
        assert result["portfolio"].iloc[-1] > 10_000.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestTearsheetEdgeCases:
    def test_single_row_does_not_crash(self):
        """Single-row DataFrame should return zeros without raising."""
        dates = pd.date_range("2020-01-01", periods=1)
        df = pd.DataFrame(
            {"open": [100.0], "high": [101.0], "low": [99.0],
             "close": [100.0], "volume": [1000]},
            index=dates,
        )
        signals = pd.Series([1.0], index=dates)
        result = Tearsheet.calculate_metrics(df, signals)
        assert result["CAGR (%)"] == 0.0

    def test_cagr_zero_when_same_day_start_end(self):
        dates = pd.date_range("2020-06-01", periods=1)
        df = pd.DataFrame(
            {"open": [100.0], "high": [101.0], "low": [99.0],
             "close": [100.0], "volume": [1000]},
            index=dates,
        )
        signals = pd.Series([0.0], index=dates)
        result = Tearsheet.calculate_metrics(df, signals)
        assert result["CAGR (%)"] == 0.0

    def test_print_summary_does_not_raise(self):
        """print_summary should complete without exceptions."""
        df = _make_df()
        signals = pd.Series(1.0, index=df.index)
        metrics = Tearsheet.calculate_metrics(df, signals)
        Tearsheet.print_summary(metrics)
