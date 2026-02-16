import time
from datetime import datetime
from .database import Database
from .fetcher import DataFetcher

class Strategy:
    def __init__(self):
        self.name = "Base Strategy"

    def on_bar(self, ticker, bar, history):
        """
        Called whenever a new bar is available.
        bar: Current bar data (OHLCV)
        history: Previous bars for context
        """
        pass

class TradingEngine:
    def __init__(self, db_path="data/stocks.db"):
        self.db = Database(db_path)
        self.fetcher = DataFetcher()

    def sync_data(self, ticker, interval, period="1mo"):
        """Downloads data and saves to DB."""
        print(f"Syncing {ticker} ({interval})...")
        df = self.fetcher.fetch_historical(ticker, interval, period=period)
        if not df.empty:
            self.db.save_data(df, ticker, interval)
            print(f"Saved {len(df)} bars.")
        else:
            print("No data found.")

    def run_backtest(self, ticker, interval, strategy, start=None, end=None):
        """Runs a strategy against historical data in the DB."""
        print(f"Starting Backtest: {ticker} ({interval})")
        df = self.db.get_data(ticker, interval, start, end)
        
        if df.empty:
            print("No local data found. Please sync first.")
            return

        for i in range(1, len(df)):
            current_bar = df.iloc[i]
            history = df.iloc[:i]
            strategy.on_bar(ticker, current_bar, history)

    def run_live(self, ticker, interval, strategy, refresh_rate=60):
        """
        Simulates live testing by fetching fresh data periodically.
        For simplicity with yfinance, we'll fetch the latest bars.
        """
        print(f"Starting Live Simulation: {ticker} ({interval})")
        while True:
            try:
                # Fetch latest data (e.g., last 5 bars to be safe)
                df = self.fetcher.fetch_historical(ticker, interval, period="1d")
                if not df.empty:
                    latest_bar = df.iloc[-1]
                    history = df.iloc[:-1]
                    strategy.on_bar(ticker, latest_bar, history)
                
                time.sleep(refresh_rate)
            except KeyboardInterrupt:
                print("Stopping live simulation...")
                break
            except Exception as e:
                print(f"Error in live simulation: {e}")
                time.sleep(refresh_rate)
