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


@patch('engine.core.config.config.FRED_API_KEY', 'dummy_test_key')
def test_fetch_macro_data_success(fetcher):
    # Setup the mock JSON response matching the official FRED API structure
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "observations": [
            {"date": "2023-01-01", "value": "4.5"},
            {"date": "2023-02-01", "value": "4.6"},
            {"date": "2023-03-01", "value": "."}  # Simulate FRED's missing data character
        ]
    }
    mock_response.raise_for_status.return_value = None

    # Patch the fetcher's cached session to return our mock
    with patch.object(fetcher.session, 'get', return_value=mock_response) as mock_get:
        df = fetcher.fetch_macro_data("UNRATE", "2023-01-01", "2023-03-01")

        # Assert the HTTP request was built correctly
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert args[0] == "https://api.stlouisfed.org/fred/series/observations"
        assert kwargs['params']['series_id'] == "UNRATE"
        assert kwargs['params']['api_key'] == "dummy_test_key"
        assert kwargs['params']['file_type'] == "json"

        # Assert the resulting DataFrame is processed correctly
        # The "." value should be coerced to NaN and dropped by _sanitize_dataframe, 
        # so we expect exactly 2 rows back, not 3.
        assert not df.empty
        assert len(df) == 2
        
        # Check column mappings and data types
        assert "indicator_name" in df.columns
        assert df["indicator_name"].iloc[0] == "UNRATE"
        assert df["value"].iloc[0] == 4.5
        assert isinstance(df["value"].iloc[0], (float, np.floating))
        assert isinstance(df["date"].iloc[0], pd.Timestamp)

@patch('engine.core.config.config.FRED_API_KEY', None)
def test_fetch_macro_data_missing_key(fetcher):
    # If the key is missing, it should immediately return an empty DataFrame 
    # without ever attempting an HTTP request.
    with patch.object(fetcher.session, 'get') as mock_get:
        df = fetcher.fetch_macro_data("UNRATE", "2023-01-01", "2023-03-01")
        
        mock_get.assert_not_called()
        assert df.empty
        
