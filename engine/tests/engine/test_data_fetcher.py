import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from engine.core.data_broker.fetcher import DataFetcher
import tenacity

@pytest.fixture
def fetcher():
    return DataFetcher()

def test_interval_mapping(fetcher):
    with patch('yfinance.download') as mock_download:
        df_mock = pd.DataFrame({
            'Open': [10, 20], 'High': [10, 20], 'Low': [10, 20], 'Close': [10, 20], 'Volume': [10, 20]
        }, index=pd.date_range('2023-01-01', periods=2))
        df_mock.index.name = 'Date'
        mock_download.return_value = df_mock
        
        fetcher.fetch_ohlcv("AAPL", "1wk", "2023-01-01", "2023-01-02")
        mock_download.assert_called_with("AAPL", start="2023-01-01", end="2023-01-02", interval="1wk", progress=False, session=fetcher.session)
        
        fetcher.fetch_ohlcv("AAPL", "1d", "2023-01-01", "2023-01-02")
        mock_download.assert_called_with("AAPL", start="2023-01-01", end="2023-01-02", interval="1d", progress=False, session=fetcher.session)
        
        fetcher.fetch_ohlcv("AAPL", "4h", "2023-01-01", "2023-01-02")
        # 4h maps to 1h
        mock_download.assert_called_with("AAPL", start="2023-01-01", end="2023-01-02", interval="1h", progress=False, session=fetcher.session)

def test_resampling_4h(fetcher):
    dates = pd.date_range('2023-01-01 00:00:00', periods=8, freq='h')
    data = {
        'Open': range(8),
        'High': range(10, 18),
        'Low': range(8),
        'Close': range(5, 13),
        'Volume': [100] * 8
    }
    df_1h = pd.DataFrame(data, index=dates)
    df_1h.index.name = 'Datetime'
    
    with patch('yfinance.download') as mock_download:
        mock_download.return_value = df_1h
        
        df_4h = fetcher.fetch_ohlcv("AAPL", "4h", "2023-01-01", "2023-01-02")
        
        assert len(df_4h) == 2
        assert df_4h['timestamp'].iloc[0] == pd.Timestamp('2023-01-01 00:00:00')
        assert df_4h['timestamp'].iloc[1] == pd.Timestamp('2023-01-01 04:00:00')
        assert df_4h['open'].iloc[0] == 0
        assert df_4h['high'].iloc[0] == 13 # max of 10, 11, 12, 13
        assert df_4h['close'].iloc[1] == 12 # last of 9, 10, 11, 12

def test_timezone_handling(fetcher):
    dates = pd.date_range('2023-01-01', periods=2, tz='America/New_York')
    df_tz = pd.DataFrame({
        'Open': [10, 20], 'High': [10, 20], 'Low': [10, 20], 'Close': [10, 20], 'Volume': [10, 20]
    }, index=dates)
    df_tz.index.name = 'Date'
    
    with patch('yfinance.download') as mock_download:
        mock_download.return_value = df_tz
        
        df = fetcher.fetch_ohlcv("AAPL", "1d", "2023-01-01", "2023-01-02")
        
        assert df['timestamp'].dt.tz is None

def test_fetch_error_handling(fetcher):
    with patch('yfinance.download') as mock_download:
        mock_download.side_effect = Exception("API Down")
        
        with pytest.raises(tenacity.RetryError):
            fetcher.fetch_ohlcv("AAPL", "1d", "2023-01-01", "2023-01-02")


@patch('engine.core.data_broker.fetcher.config.FRED_API_KEY', 'dummy_test_key')
def test_fetch_macro_data_success(fetcher):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "observations": [
            {"date": "2023-01-01", "value": "4.5"},
            {"date": "2023-02-01", "value": "4.6"},
            {"date": "2023-03-01", "value": "."}  # FRED's missing-data sentinel
        ]
    }
    mock_response.raise_for_status.return_value = None

    with patch('engine.core.data_broker.fetcher.requests.get', return_value=mock_response) as mock_get:
        df = fetcher.fetch_macro_data("UNRATE", "2023-01-01", "2023-03-01")

        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert args[0] == "https://api.stlouisfed.org/fred/series/observations"
        assert kwargs['params']['series_id'] == "UNRATE"
        assert kwargs['params']['api_key'] == "dummy_test_key"
        assert kwargs['params']['file_type'] == "json"
        assert kwargs['params']['observation_start'] == "2023-01-01"
        assert kwargs['params']['observation_end'] == "2023-03-01"

        # The "." row should coerce to NaN and be dropped by _sanitize_dataframe.
        assert len(df) == 2
        assert "indicator_name" in df.columns
        assert df["indicator_name"].iloc[0] == "UNRATE"
        assert df["value"].iloc[0] == 4.5
        assert isinstance(df["value"].iloc[0], (float, np.floating))
        assert isinstance(df["date"].iloc[0], pd.Timestamp)


@patch('engine.core.data_broker.fetcher.config.FRED_API_KEY', None)
def test_fetch_macro_data_missing_key(fetcher):
    with patch('engine.core.data_broker.fetcher.requests.get') as mock_get:
        df = fetcher.fetch_macro_data("UNRATE", "2023-01-01", "2023-03-01")
        mock_get.assert_not_called()
        assert df.empty


@patch('engine.core.data_broker.fetcher.config.FRED_API_KEY', 'dummy_test_key')
def test_fetch_macro_data_empty_observations(fetcher):
    mock_response = MagicMock()
    mock_response.json.return_value = {"observations": []}
    mock_response.raise_for_status.return_value = None
    with patch('engine.core.data_broker.fetcher.requests.get', return_value=mock_response):
        df = fetcher.fetch_macro_data("UNRATE", "2023-01-01", "2023-03-01")
        assert df.empty


@patch('engine.core.data_broker.fetcher.config.FRED_API_KEY', 'dummy_test_key')
def test_fetch_macro_data_http_error(fetcher):
    import requests as _requests
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = _requests.HTTPError("400 Bad Request")
    with patch('engine.core.data_broker.fetcher.requests.get', return_value=mock_response):
        df = fetcher.fetch_macro_data("UNRATE", "2023-01-01", "2023-03-01")
        # Errors are swallowed and converted to empty DataFrames (by design).
        assert df.empty


@pytest.mark.skipif(
    os.getenv("RUN_LIVE_FRED") != "1" or not os.getenv("FRED_API_KEY"),
    reason="Live FRED test. Set RUN_LIVE_FRED=1 and FRED_API_KEY to enable.",
)
def test_fetch_macro_data_live(fetcher):
    """Hits the real FRED API. Gated by RUN_LIVE_FRED=1 to keep CI offline-safe."""
    df = fetcher.fetch_macro_data("UNRATE", "2024-01-01", "2024-06-01")
    assert not df.empty, "Live FRED call returned no rows — check FRED_API_KEY format (32 lowercase hex chars)."
    assert set(df.columns) >= {"date", "value", "indicator_name"}
    assert df["indicator_name"].iloc[0] == "UNRATE"
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert pd.api.types.is_numeric_dtype(df["value"])

