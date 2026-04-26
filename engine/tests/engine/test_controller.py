import pytest
from pydantic import ValidationError
from datetime import datetime
from engine.core.controller import ApplicationController, ExecutionMode, MultiAssetMode
from engine.core.exceptions import ValidationError as AppValidationError, StrategyError
from unittest.mock import patch, MagicMock, call
import pandas as pd

def test_payload_validation_failure():
    controller = ApplicationController()
    # Missing required field 'mode'
    payload = {
        "strategy": "dummy",
        "assets": ["AAPL"],
        "interval": "1d"
    }
    
    with pytest.raises(AppValidationError, match="Invalid job payload"):
        controller.execute_job(payload)

@patch("engine.core.controller.ApplicationController._handle_backtest")
@patch("os.path.exists")
def test_route_backtest_dispatch(mock_exists, mock_handle_backtest):
    mock_exists.return_value = True
    controller = ApplicationController()
    
    payload = {
        "strategy": "dummy",
        "assets": ["AAPL", "MSFT"],
        "interval": "1d",
        "mode": ExecutionMode.BACKTEST,
        "multi_asset_mode": "BATCH"
    }
    
    controller.execute_job(payload)
    
    # Assert _handle_backtest was called correctly
    mock_handle_backtest.assert_called_once()
    args, kwargs = mock_handle_backtest.call_args
    assert "AAPL" in args[1]
    assert "MSFT" in args[1]

@patch("engine.core.controller.LocalBacktester")
@patch("os.path.exists")
def test_route_signal_only_timestamp(mock_exists, mock_backtester, mock_data_broker):
    mock_exists.return_value = True
    
    # Mock backtester output
    mock_instance = MagicMock()
    # Create a mock Series with a proper DatetimeIndex
    index = pd.date_range("2023-01-01", periods=3)
    mock_series = pd.Series([0.5, -0.5, 1.0], index=index)
    mock_instance.run_batch.return_value = {"AAPL": mock_series}
    mock_backtester.return_value = mock_instance
    
    controller = ApplicationController()
    
    payload = {
        "strategy": "dummy",
        "assets": ["AAPL"],
        "interval": "1d",
        "mode": ExecutionMode.SIGNAL_ONLY,
    }
    
    result = controller.execute_job(payload)
    assert "AAPL" in result
    aapl_res = result["AAPL"]
    
    assert aapl_res["signal"] == 1.0
    assert aapl_res["mode"] == "SIGNAL_ONLY"
    assert aapl_res["asset"] == "AAPL"
    
    # Verify timestamp is ISO format
    try:
        datetime.fromisoformat(aapl_res["timestamp"])
    except ValueError:
        pytest.fail("Timestamp is not in valid ISO format")


# ---------------------------------------------------------------------------
# Strategy-not-found guard
# ---------------------------------------------------------------------------

def test_strategy_not_found_raises():
    controller = ApplicationController(strategies_dir="/nonexistent/dir")
    payload = {
        "strategy": "missing_strat",
        "assets": ["AAPL"],
        "interval": "1d",
        "mode": ExecutionMode.BACKTEST,
    }
    with pytest.raises(StrategyError, match="not found"):
        controller.execute_job(payload)


def test_unsupported_mode_raises():
    """A payload with an invalid mode string should raise AppValidationError."""
    controller = ApplicationController()
    payload = {
        "strategy": "dummy",
        "assets": ["AAPL"],
        "interval": "1d",
        "mode": "INVALID_MODE",
    }
    with pytest.raises(AppValidationError, match="Invalid job payload"):
        controller.execute_job(payload)


# ---------------------------------------------------------------------------
# _handle_backtest — empty data returns empty dict
# ---------------------------------------------------------------------------

@patch("engine.core.controller.DataBroker.get_data")
@patch("os.path.exists")
def test_handle_backtest_empty_data_returns_empty(mock_exists, mock_get_data):
    mock_exists.return_value = True
    mock_get_data.return_value = pd.DataFrame()  # empty — simulates no data

    controller = ApplicationController()
    payload = {
        "strategy": "dummy",
        "assets": ["AAPL"],
        "interval": "1d",
        "mode": ExecutionMode.BACKTEST,
    }
    result = controller.execute_job(payload)
    assert result == {}


# ---------------------------------------------------------------------------
# _handle_backtest — batch partial failure (one asset fails)
# ---------------------------------------------------------------------------

@patch("engine.core.controller.LocalBacktester")
@patch("engine.core.controller.DataBroker.get_data")
@patch("os.path.exists")
def test_handle_backtest_partial_failure(mock_exists, mock_get_data, mock_backtester):
    mock_exists.return_value = True

    index = pd.date_range("2023-01-01", periods=3)
    good_df = pd.DataFrame(
        {"Open": [100.0, 101.0, 102.0], "High": [103.0]*3, "Low": [98.0]*3,
         "Close": [101.0, 102.0, 103.0], "Volume": [1000]*3},
        index=index,
    )
    mock_get_data.return_value = good_df

    # backtester.run_batch returns an empty Series for the failing ticker
    good_signals = pd.Series([1.0, -1.0, 0.5], index=index)
    empty_signals = pd.Series(dtype=float)

    mock_instance = MagicMock()
    mock_instance.run_batch.return_value = {
        "AAPL": good_signals,
        "TSLA": empty_signals,
    }
    mock_backtester.return_value = mock_instance

    controller = ApplicationController()
    payload = {
        "strategy": "dummy",
        "assets": ["AAPL", "TSLA"],
        "interval": "1d",
        "mode": ExecutionMode.BACKTEST,
    }
    result = controller.execute_job(payload)

    assert "AAPL" in result
    assert "TSLA" in result
    # AAPL should have metrics; TSLA should have an error marker
    assert "error" in result["TSLA"]


# ---------------------------------------------------------------------------
# _handle_signal_only — no data returns error dict
# ---------------------------------------------------------------------------

@patch("engine.core.controller.DataBroker.get_data")
@patch("os.path.exists")
def test_handle_signal_only_no_data_returns_error(mock_exists, mock_get_data):
    mock_exists.return_value = True
    mock_get_data.return_value = pd.DataFrame()

    controller = ApplicationController()
    payload = {
        "strategy": "dummy",
        "assets": ["AAPL"],
        "interval": "1d",
        "mode": ExecutionMode.SIGNAL_ONLY,
    }
    result = controller.execute_job(payload)
    assert "AAPL" in result
    assert "error" in result["AAPL"]


# ---------------------------------------------------------------------------
# _handle_signal_only — single string asset normalized to list
# ---------------------------------------------------------------------------

@patch("engine.core.controller.LocalBacktester")
@patch("engine.core.controller.DataBroker.get_data")
@patch("os.path.exists")
def test_handle_signal_only_single_string_asset(mock_exists, mock_get_data,
                                                 mock_backtester, mock_data_broker):
    """_handle_signal_only accepts a single string asset without crashing."""
    mock_exists.return_value = True

    index = pd.date_range("2023-01-01", periods=3)
    mock_get_data.return_value = pd.DataFrame(
        {"Open": [100.0, 101.0, 102.0], "High": [103.0]*3, "Low": [98.0]*3,
         "Close": [101.0, 102.0, 103.0], "Volume": [1000]*3},
        index=index,
    )

    mock_instance = MagicMock()
    signals = pd.Series([0.5, -0.5, 1.0], index=index)
    mock_instance.run_batch.return_value = {"AAPL": signals}
    mock_backtester.return_value = mock_instance

    controller = ApplicationController()
    # Call _handle_signal_only directly with a raw string
    result = controller._handle_signal_only("dummy_path", "AAPL", "1d", None, None)

    assert "AAPL" in result
    assert result["AAPL"]["signal"] == pytest.approx(1.0)
