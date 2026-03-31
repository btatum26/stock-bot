import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Tuple
from sqlalchemy import select, func
from sqlalchemy.orm import Session

from .database import Database, OHLCV
from .fetcher import DataFetcher

class DataBroker:
    def __init__(self):
        self.db = Database()
        self.fetcher = DataFetcher()

    def _get_padding(self, start_date: datetime, interval: str, periods: int = 200) -> datetime:
        """Calculates rough calendar padding to ensure indicator warm-up."""
        if interval == '1d':
            return start_date - timedelta(days=periods * 1.4)
        elif interval == '1wk':
            return start_date - timedelta(weeks=periods)
        elif interval == '4h':
            return start_date - timedelta(days=(periods / 2))
        return start_date

    def _get_db_bounds(self, ticker: str, interval: str) -> Tuple[Optional[datetime], Optional[datetime]]:
        with Session(self.db.engine) as session:
            stmt = select(func.min(OHLCV.timestamp), func.max(OHLCV.timestamp)).where(
                OHLCV.ticker == ticker,
                OHLCV.interval == interval
            )
            result = session.execute(stmt).first()
            return result[0], result[1]

    # Yahoo Finance hard limits on how far back each interval can reach
    _YF_MAX_HISTORY: dict = {
        '1m':  timedelta(days=7),
        '2m':  timedelta(days=60),
        '5m':  timedelta(days=60),
        '15m': timedelta(days=60),
        '30m': timedelta(days=60),
        '60m': timedelta(days=730),
        '1h':  timedelta(days=730),
        '90m': timedelta(days=60),
    }

    def get_data(self, ticker: str, interval: str, start: datetime, end: datetime) -> pd.DataFrame:
        """The primary interface for the strategy engine."""

        # Clamp start date to Yahoo Finance's per-interval history limit
        max_lookback = self._YF_MAX_HISTORY.get(interval)
        if max_lookback is not None:
            earliest_allowed = end - max_lookback
            if start < earliest_allowed:
                start = earliest_allowed

        padded_start = self._get_padding(start, interval, periods=200)

        # Re-apply the cap after padding (padding may push it back past the limit)
        if max_lookback is not None:
            earliest_allowed = end - max_lookback
            if padded_start < earliest_allowed:
                padded_start = earliest_allowed

        # 15-Minute Rule: Fetch directly, bypass DB
        if interval == '15m':
            print(f"[{ticker}] 15m requested. Bypassing DB.")
            df = self.fetcher.fetch_ohlcv(
                ticker, interval, 
                start=padded_start.strftime('%Y-%m-%d'), 
                end=(end + timedelta(days=1)).strftime('%Y-%m-%d')
            )
            if not df.empty:
                df.set_index('timestamp', inplace=True)
            return df

        db_min, db_max = self._get_db_bounds(ticker, interval)
        needs_fetch = False

        if not db_min or not db_max:
            needs_fetch = True
            fetch_start, fetch_end = padded_start, end
        elif padded_start < db_min or end > db_max:
            needs_fetch = True
            fetch_start, fetch_end = padded_start, end

        if needs_fetch:
            df_new = self.fetcher.fetch_ohlcv(
                ticker, interval, 
                start=fetch_start.strftime('%Y-%m-%d'), 
                end=(fetch_end + timedelta(days=1)).strftime('%Y-%m-%d')
            )
            
            if not df_new.empty:
                df_new['ticker'] = ticker
                df_new['interval'] = interval
                records = df_new.to_dict('records')
                
                with Session(self.db.engine) as session:
                    try:
                        from sqlalchemy.dialects.sqlite import insert
                        stmt = insert(OHLCV).values(records)
                        stmt = stmt.on_conflict_do_nothing(index_elements=['ticker', 'timestamp', 'interval'])
                        session.execute(stmt)
                        session.commit()
                        print(f"[{ticker}] Hydrated DB with {len(records)} new {interval} bars.")
                    except Exception as e:
                        session.rollback()
                        print(f"[{ticker}] DB Insert Failed: {e}")

        # Serve from DB
        with Session(self.db.engine) as session:
            stmt = select(OHLCV).where(
                OHLCV.ticker == ticker,
                OHLCV.interval == interval,
                OHLCV.timestamp >= padded_start,
                OHLCV.timestamp <= end
            ).order_by(OHLCV.timestamp.asc())
            
            df_final = pd.read_sql(stmt, session.bind)
            if not df_final.empty:
                df_final.set_index('timestamp', inplace=True)
                
        return df_final[['open', 'high', 'low', 'close', 'volume']]