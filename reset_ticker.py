from src.database import Database, OHLCV
import sys

def reset_ticker(ticker, interval):
    print(f"Resetting data for {ticker} ({interval})...")
    db = Database("data/stocks.db")
    session = db.Session()
    
    # Delete all records for this ticker/interval
    deleted = session.query(OHLCV).filter_by(
        ticker=ticker, 
        interval=interval
    ).delete()
    
    session.commit()
    print(f"Deleted {deleted} records.")
    session.close()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python reset_ticker.py <TICKER> <INTERVAL>")
        print("Example: python reset_ticker.py AAPL 4h")
    else:
        reset_ticker(sys.argv[1], sys.argv[2])
