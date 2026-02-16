import argparse
from src.engine import TradingEngine, Strategy
import pandas as pd

class SMACrossover(Strategy):
    def __init__(self, short_window=5, long_window=20):
        super().__init__()
        self.name = "SMA Crossover"
        self.short_window = short_window
        self.long_window = long_window

    def on_bar(self, ticker, bar, history):
        if len(history) < self.long_window:
            return

        # Calculate SMAs
        short_sma = history['Close'].tail(self.short_window).mean()
        long_sma = history['Close'].tail(self.long_window).mean()
        
        # Check for crossover
        if short_sma > long_sma:
            print(f"[{bar.name}] BUY Signal for {ticker} at {bar['Close']:.2f} (SMA {short_sma:.2f} > {long_sma:.2f})")
        elif short_sma < long_sma:
            print(f"[{bar.name}] SELL Signal for {ticker} at {bar['Close']:.2f} (SMA {short_sma:.2f} < {long_sma:.2f})")

def main():
    parser = argparse.ArgumentParser(description="Stock Trading Bot Tool")
    parser.add_argument("--mode", choices=["sync", "backtest", "live", "snapshot"], required=True, help="Mode to run")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Ticker symbol")
    parser.add_argument("--interval", choices=["1h", "4h", "1d"], default="1d", help="Data interval")
    parser.add_argument("--period", type=str, default="1y", help="Period for sync (e.g. 1y, 1mo)")

    args = parser.parse_args()
    engine = TradingEngine()

    if args.mode == "sync":
        engine.sync_data(args.ticker, args.interval, period=args.period)
    
    elif args.mode == "snapshot":
        from src.snapshot import DataSnapshot
        snapshot = DataSnapshot(engine)
        snapshot.run(interval=args.interval, period="10y")

    elif args.mode == "backtest":
        strategy = SMACrossover()
        engine.run_backtest(args.ticker, args.interval, strategy)

    elif args.mode == "live":
        strategy = SMACrossover()
        # For live simulation, we'll check every 5 minutes if using 1h bars, 
        # but for 1d bars maybe just check once.
        refresh = 60 if args.interval != "1d" else 3600
        engine.run_live(args.ticker, args.interval, strategy, refresh_rate=refresh)

if __name__ == "__main__":
    main()
