import pytest
from pydantic import ValidationError
from datetime import datetime
from engine.core.controller import ApplicationController, ExecutionMode
from engine.core.exceptions import ValidationError as AppValidationError
from unittest.mock import patch, MagicMock
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
