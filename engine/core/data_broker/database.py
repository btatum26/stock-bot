import os
from datetime import datetime
from typing import Optional
import pandas as pd
from sqlalchemy import (
    create_engine, Column, String, Float, DateTime, Integer,
    UniqueConstraint, Index, text, event
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.pool import NullPool
from ..logger import data_logger as logger

Base = declarative_base()

class OHLCV(Base):
    __tablename__ = 'ohlcv'
    
    id = Column(Integer, primary_key=True)
    ticker = Column(String, index=True)
    timestamp = Column(DateTime, index=True)
    interval = Column(String)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)

    __table_args__ = (
        UniqueConstraint('ticker', 'timestamp', 'interval', name='_ticker_ts_interval_uc'),
        Index('idx_ticker_ts_interval', 'ticker', 'timestamp', 'interval'),
    )

class FetchLedger(Base):
    """Tracks (ticker, interval) ranges confirmed to have no upstream data.

    When a backward-gap fetch succeeds but returns nothing, we record
    `empty_before` so subsequent requests skip the redundant round-trip.
    """
    __tablename__ = 'fetch_ledger'

    ticker       = Column(String,   primary_key=True)
    interval     = Column(String,   primary_key=True)
    empty_before = Column(DateTime, nullable=False)


class Database:
    def __init__(self, db_path="data/stocks.db"):
        try:
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            
            # 1. Added NullPool to prevent file locking on local systems
            self.engine = create_engine(f"sqlite:///{db_path}", poolclass=NullPool)
            
            # 2. Attach the WAL mode PRAGMA strictly to this engine instance
            event.listen(self.engine, "connect", self._set_sqlite_pragma)
            
            Base.metadata.create_all(self.engine)
            self.Session = sessionmaker(bind=self.engine)
        except Exception as e:
            logger.error(f"Database initialization error: {e}", exc_info=True)
            raise

    def _set_sqlite_pragma(self, dbapi_connection, connection_record):
        """Enforces WAL mode for concurrent reads/writes."""
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA synchronous=NORMAL;")
        cursor.execute("PRAGMA foreign_keys=ON;")
        cursor.close()

    def save_data(self, df, ticker, interval):
        """Saves a pandas DataFrame to the database using an efficient bulk operation."""
        if df.empty:
            return

        save_df = df.copy()
        save_df['ticker'] = ticker
        save_df['interval'] = interval
        save_df = save_df.reset_index()
        
        ts_col = next((c for c in save_df.columns if c.lower() in ['date', 'datetime', 'timestamp']), None)
        if ts_col:
            save_df = save_df.rename(columns={ts_col: 'timestamp'})
        
        column_mapping = {
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }
        save_df = save_df.rename(columns=column_mapping)
        
        required_cols = ['ticker', 'timestamp', 'interval', 'open', 'high', 'low', 'close', 'volume']
        save_df = save_df[required_cols]

        try:
            with self.engine.begin() as conn:
                save_df.to_sql('temp_ohlcv', conn, if_exists='replace', index=False)
                
                insert_stmt = text("""
                    INSERT OR IGNORE INTO ohlcv (ticker, timestamp, interval, open, high, low, close, volume)
                    SELECT ticker, timestamp, interval, open, high, low, close, volume FROM temp_ohlcv
                """)
                conn.execute(insert_stmt)
                
                conn.execute(text("DROP TABLE temp_ohlcv"))
        except Exception as e:
            logger.error(f"Error bulk saving data: {e}", exc_info=True)
            raise

    def get_latest_timestamp(self, ticker, interval):
        """Returns the most recent timestamp for a ticker/interval or None."""
        session = self.Session()
        try:
            result = session.query(OHLCV.timestamp).filter_by(
                ticker=ticker, 
                interval=interval
            ).order_by(OHLCV.timestamp.desc()).first()
            return result[0] if result else None
        finally:
            session.close()

    def get_all_tickers(self):
        """Returns a list of all distinct tickers in the database."""
        session = self.Session()
        try:
            results = session.query(OHLCV.ticker).distinct().all()
            return [r[0] for r in results]
        finally:
            session.close()

    def get_data(self, ticker, interval, start=None, end=None):
        """Retrieves data from the database as a pandas DataFrame."""
        try:
            session = self.Session()
            query = session.query(OHLCV).filter_by(ticker=ticker, interval=interval)
            if start:
                query = query.filter(OHLCV.timestamp >= start)
            if end:
                query = query.filter(OHLCV.timestamp <= end)
            
            # Vectorized Read: Bypassing ORM loops for speed
            df = pd.read_sql_query(query.statement, self.engine)
            session.close()

            if df.empty:
                return pd.DataFrame()

            # Clean up column names to match system conventions
            column_mapping = {
                'timestamp': 'Timestamp',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }
            df = df.rename(columns=column_mapping)
            df.set_index('Timestamp', inplace=True)
            
            # Ensure the index is a DatetimeIndex
            df.index = pd.to_datetime(df.index)
            
            # Select only the relevant OHLCV columns
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
        except Exception as e:
            logger.error(f"Error retrieving data from database: {e}", exc_info=True)
            return pd.DataFrame()

    def get_empty_before(self, ticker: str, interval: str) -> Optional[datetime]:
        """Returns the `empty_before` sentinel for (ticker, interval), or None."""
        with Session(self.engine) as session:
            row = session.get(FetchLedger, (ticker, interval))
            return row.empty_before if row else None

    def set_empty_before(self, ticker: str, interval: str, empty_before: datetime) -> None:
        """Upserts the `empty_before` sentinel for (ticker, interval)."""
        with Session(self.engine) as session:
            try:
                from sqlalchemy.dialects.sqlite import insert
                stmt = (
                    insert(FetchLedger)
                    .values(ticker=ticker, interval=interval, empty_before=empty_before)
                    .on_conflict_do_update(
                        index_elements=['ticker', 'interval'],
                        set_={'empty_before': empty_before},
                    )
                )
                session.execute(stmt)
                session.commit()
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to update fetch ledger for {ticker}/{interval}: {e}", exc_info=True)