import argparse
from src.engine import TradingEngine

def main():
    parser = argparse.ArgumentParser(description="Sync individual ticker data")
    parser.add_argument("--ticker", type=str, required=True, help="Ticker symbol")
    parser.add_argument("--interval", choices=["1w", "1d", "4h", "1h", "30m", "15m"], default="1d", help="Data interval")
    parser.add_argument("--period", type=str, default="1y", help="Period for sync (e.g. 1y, 1mo)")
    
    args = parser.parse_args()
    engine = TradingEngine()
    
    try:
        engine.sync_data(args.ticker, args.interval, period=args.period)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
