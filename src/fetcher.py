import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

class DataFetcher:
    @staticmethod
    def fetch_historical(ticker, interval, period="max", start=None, end=None):
        """
        Fetches historical data from yfinance.
        """
        yf_interval = interval
        if interval == "4h":
            yf_interval = "1h"
        
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(interval=yf_interval, period=period, start=start, end=end)
            
            if df is None or df.empty:
                print(f"Warning: No data found for {ticker} at interval {interval}.")
                return pd.DataFrame()

            if interval == "4h":
                # Resample 1h data to 4h
                df = df.resample('4h').agg({
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
