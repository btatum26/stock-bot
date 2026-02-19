import time
from datetime import datetime
from .database import Database
from .fetcher import DataFetcher
from .features.signals import SignalEvent
from typing import List, Dict, Any, Optional
import pandas as pd

class SignalEvaluation:
    """
    Tracks and evaluates the performance of signals generated during a backtest.
    """
    def __init__(self, forward_window=5, threshold=0.01):
        self.forward_window = forward_window
        self.threshold = threshold
        self.total_signals = 0
        self.correct_calls = 0
        self.incorrect_calls = 0
        self.results = [] # List of {timestamp, side, entry_price, exit_price, pnl, success}

    def evaluate(self, df: pd.DataFrame, event: SignalEvent):
        iloc = event.index
        if iloc + self.forward_window >= len(df):
            return None # Not enough future data to evaluate
        
        entry_price = event.value
        future_prices = df['Close'].iloc[iloc+1 : iloc+1+self.forward_window]
        
        # Simple success metric: Did price move in our direction within the window?
        if event.side == 'buy':
            max_forward = future_prices.max()
            pnl = (max_forward - entry_price) / entry_price
            success = pnl >= self.threshold
        else: # sell
            min_forward = future_prices.min()
            pnl = (entry_price - min_forward) / entry_price
            success = pnl >= self.threshold

        self.total_signals += 1
        if success: self.correct_calls += 1
        else: self.incorrect_calls += 1

        res = {
            "timestamp": event.timestamp,
            "side": event.side,
            "entry": entry_price,
            "max_fwd_pnl": pnl,
            "success": success,
            "model": event.name
        }
        self.results.append(res)
        return res

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

    def sync_data(self, ticker, interval, period="1mo", quiet=False):
        """Downloads data and saves to DB."""
        if not quiet: print(f"\nSyncing {ticker} ({interval})...")
        df = self.fetcher.fetch_historical(ticker, interval, period=period)
        if not df.empty:
            self.db.save_data(df, ticker, interval)
            if not quiet: print(f"Saved {len(df)} bars.")
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
