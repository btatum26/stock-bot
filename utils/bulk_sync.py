import argparse
from src.engine import TradingEngine
from src.snapshot import DataSnapshot

def main():
    parser = argparse.ArgumentParser(description="Bulk data collection (snapshot)")
    parser.add_argument("--period", type=str, default="10y", help="Period for historical data (e.g. 10y)")
    
    args = parser.parse_args()
    engine = TradingEngine()
    snapshot = DataSnapshot(engine)
    
    try:
        snapshot.run(period=args.period)
    except KeyboardInterrupt:
        print("\nBulk operation cancelled by user.")
    except Exception as e:
        print(f"Error during bulk sync: {e}")

if __name__ == "__main__":
    main()
