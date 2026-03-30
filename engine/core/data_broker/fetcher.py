import logging
import yfinance as yf
import pandas as pd
import requests
import requests_cache
from datetime import datetime
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from ..config import config

# 1. Anomaly Logging
logging.basicConfig(
    filename='data/ingestion.log', 
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 2. Local Disk Caching
session = requests_cache.CachedSession('yfinance.cache', expire_after=43200)

class DataFetcher:
    def __init__(self, av_api_key: str = "YOUR_FREE_KEY"):
        self.session = session
        self.av_api_key = av_api_key
        self.FRED_API_KEY = config.FRED_API_KEY  # Access FRED API key from config

    def _sanitize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """API Data Sanitization Layer: Drop NaNs and invalid rows."""
        original_len = len(df)
        df = df.dropna()
        if len(df) < original_len:
            logging.warning(f"Sanitization dropped {original_len - len(df)} rows containing NaN.")
        return df
    

    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    def fetch_ohlcv(self, ticker: str, interval: str, start: str, end: str) -> pd.DataFrame:
        """Fetches OHLCV data with backoff retries."""
        try:
            yf_interval = interval if interval != '4h' else '1h' 
            df = yf.download(ticker, start=start, end=end, interval=yf_interval, progress=False, session=self.session)

            if df.empty:
                return pd.DataFrame()

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
                
            df = df.reset_index()
            ts_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
            df.rename(columns={
                ts_col: 'timestamp', 'Open': 'open', 'High': 'high', 
                'Low': 'low', 'Close': 'close', 'Volume': 'volume'
            }, inplace=True)

            if df['timestamp'].dt.tz is not None:
                df['timestamp'] = df['timestamp'].dt.tz_localize(None)

            if interval == '4h' and not df.empty:
                df.set_index('timestamp', inplace=True)
                df = df.resample('4h', closed='left', label='left').agg({
                    'open': 'first', 'high': 'max', 'low': 'min', 
                    'close': 'last', 'volume': 'sum'
                }).dropna().reset_index()

            return self._sanitize_dataframe(df[['timestamp', 'open', 'high', 'low', 'close', 'volume']])
        
        except Exception as e:
            logging.error(f"Failed to fetch OHLCV for {ticker}: {e}")
            raise e

    @retry(wait=wait_exponential(multiplier=2, min=4, max=10), stop=stop_after_attempt(3))
    def fetch_fundamentals(self, ticker: str) -> dict:
        """Fetches fundamentals from YF, falls back to Alpha Vantage on failure."""
        try:
            stock = yf.Ticker(ticker, session=self.session)
            info = stock.info
            # Extract key ingredients
            return {
                "eps": info.get("trailingEps"),
                "book_value": info.get("bookValue"),
                "total_debt": info.get("totalDebt"),
                "total_cash": info.get("totalCash")
            }
        except Exception as e:
            logging.warning(f"YFinance fundamentals failed for {ticker}. Attempting Alpha Vantage fallback. Error: {e}")
            return self._alpha_vantage_fallback(ticker)

    def _alpha_vantage_fallback(self, ticker: str) -> dict:
        """Fallback for fundamentals if YF blocks us."""
        url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={self.av_api_key}"
        try:
            response = requests.get(url).json()
            # Note: Alpha Vantage returns strings, must cast to float
            return {
                "eps": float(response.get("EPS", 0)),
                "book_value": float(response.get("BookValue", 0)),
                "total_debt": None, # AV Overview doesn't give direct total debt without balance sheet endpoint
                "total_cash": None
            }
        except Exception as e:
            logging.error(f"Alpha Vantage fallback failed for {ticker}: {e}")
            return {}

    def fetch_macro_data(self, indicator: str, start: str, end: str) -> pd.DataFrame:
        """Fetches Macro data directly from the official FRED API."""
        if not config.FRED_API_KEY:
            logging.error("FRED API key not configured. Check your .env and config.")
            return pd.DataFrame()

        # Official FRED API Endpoint for series observations
        url = "https://api.stlouisfed.org/fred/series/observations"
        
        params = {
            "series_id": indicator,
            "api_key": config.FRED_API_KEY,
            "file_type": "json",
            "observation_start": start,
            "observation_end": end,
        }

        try:
            # Using your existing cached requests session!
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if "observations" not in data or not data["observations"]:
                logging.warning(f"No FRED observations found for {indicator}.")
                return pd.DataFrame()

            # Convert JSON observations to a pandas DataFrame
            df = pd.DataFrame(data["observations"])
            
            # Keep only the relevant columns and map them to your system's schema
            df = df[['date', 'value']]
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df['indicator_name'] = indicator
            
            return self._sanitize_dataframe(df)

        except Exception as e:
            logging.error(f"Failed to fetch direct FRED data for {indicator}: {e}")
            return pd.DataFrame()