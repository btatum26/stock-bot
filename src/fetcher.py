import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

class DataFetcher:
    @staticmethod
    def fetch_historical(ticker, interval, period="max", start=None, end=None):
        """
        Fetches historical data from yfinance.
        Intervals: 1w, 1d, 4h, 1h, 30m, 15m
        """
        # Map our internal interval to yfinance interval
        mapping = {
            "1w": "1wk",
            "1d": "1d",
            "4h": "1h",
            "1h": "1h",
            "30m": "30m",
            "15m": "15m"
        }
        yf_interval = mapping.get(interval, "1d")
        
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(interval=yf_interval, period=period, start=start, end=end)
            
            if df is None or df.empty:
                return pd.DataFrame()

            if interval == "4h":
                # Resample 1h data to 4h
                # We need to ensure the bins align with market open (typically 9:30 AM ET)
                # Using origin='start_day' aligns to midnight, so 9:30 is 9.5h offset.
                # However, a simpler way for stocks is to just resample and drop incomplete/empty bins.
                # But to avoid "shifting" bars based on start date, we must fix the origin.
                # 'start' ensures the first bin starts at the first timestamp in the data.
                # This is safer for consistent backtesting if we always fetch from the same start.
                # But for incremental updates, we want fixed alignment.
                # Let's align to the epoch (default) but ensure we handle timezone consistently.
                
                # Standard approach: resample with a closed 'left' and label 'left'
                df = df.resample('4h', origin='start_day', closed='left', label='left').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()
                
            return df
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return pd.DataFrame()

    @staticmethod
    def fetch_live_quote(ticker):
        """Fetches the latest available price."""
        try:
            stock = yf.Ticker(ticker)
            return stock.fast_info['last_price']
        except Exception as e:
            print(f"Error fetching live quote for {ticker}: {e}")
            return None
