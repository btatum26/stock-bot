from src.database import Database, OHLCV
from sqlalchemy import func

def clean_db():
    print("Cleaning database duplicates...")
    db = Database("data/stocks.db")
    session = db.Session()
    
    # Identify duplicates: (ticker, timestamp, interval) having count > 1
    duplicates = session.query(
        OHLCV.ticker, OHLCV.timestamp, OHLCV.interval, func.count('*')
    ).group_by(
        OHLCV.ticker, OHLCV.timestamp, OHLCV.interval
    ).having(func.count('*') > 1).all()
    
    print(f"Found {len(duplicates)} duplicate groups.")
    
    deleted_count = 0
    for t, ts, interval, count in duplicates:
        # Get all records for this group
        records = session.query(OHLCV).filter_by(
            ticker=t, timestamp=ts, interval=interval
        ).order_by(OHLCV.id).all()
        
        # Keep the first (lowest ID), delete the rest
        for r in records[1:]:
            session.delete(r)
            deleted_count += 1
            
    session.commit()
    print(f"Removed {deleted_count} duplicate records.")
    session.close()

if __name__ == "__main__":
    clean_db()
