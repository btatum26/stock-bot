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

    # Earliest date to request from Yahoo for unlimited intervals (1d, 1wk).
    # yfinance will simply return whatever data exists from this point forward.
    _EPOCH_START = datetime(1900, 1, 1)

    def get_data(self, ticker: str, interval: str, start: Optional[datetime] = None,
                 end: Optional[datetime] = None) -> pd.DataFrame:
        """The primary interface for the strategy engine.

        For daily/weekly intervals, start is ignored — all available history is
        fetched and cached, then the full dataset is returned so the GUI can
        control the view window independently.

        For intraday intervals with Yahoo history limits, start is clamped to
        the maximum allowed lookback.
        """
        if end is None:
            end = datetime.utcnow()

        # Clamp intraday start dates to Yahoo Finance's hard per-interval limits
        max_lookback = self._YF_MAX_HISTORY.get(interval)
        if max_lookback is not None:
            # Intraday: honour the caller's start but don't exceed Yahoo's cap
            if start is None:
                start = end - max_lookback
            else:
                earliest_allowed = end - max_lookback
                if start < earliest_allowed:
                    start = earliest_allowed
        else:
            # Daily / weekly: always fetch full history
            start = self._EPOCH_START

        # 15-Minute Rule: Fetch directly, bypass DB (too short-lived to cache)
        if interval == '15m':
            print(f"[{ticker}] 15m requested. Bypassing DB.")
            df = self.fetcher.fetch_ohlcv(
                ticker, interval,
                start=start.strftime('%Y-%m-%d'),
                end=(end + timedelta(days=1)).strftime('%Y-%m-%d')
            )
            if not df.empty:
                df.set_index('timestamp', inplace=True)
            return df

        db_min, db_max = self._get_db_bounds(ticker, interval)

        # Fetch when: no cached data at all, history extends earlier than DB,
        # or DB is stale (today's date is beyond the last cached bar).
        today = datetime.utcnow().date()
        db_max_date = db_max.date() if db_max else None
        needs_fetch = (
            not db_min or not db_max
            or start < db_min
            or db_max_date < today
        )

        if needs_fetch:
            df_new = self.fetcher.fetch_ohlcv(
                ticker, interval,
                start=start.strftime('%Y-%m-%d'),
                end=(end + timedelta(days=1)).strftime('%Y-%m-%d')
            )

            if not df_new.empty:
                df_new['ticker'] = ticker
                df_new['interval'] = interval
                records = df_new.to_dict('records')

                # SQLite limit: 999 bound parameters per statement.
                # Each OHLCV row has 8 columns → max 124 rows per batch.
                _COLS_PER_ROW = 8
                _BATCH_SIZE = 999 // _COLS_PER_ROW  # 124

                with Session(self.db.engine) as session:
                    try:
                        from sqlalchemy.dialects.sqlite import insert
                        for i in range(0, len(records), _BATCH_SIZE):
                            batch = records[i:i + _BATCH_SIZE]
                            stmt = insert(OHLCV).values(batch)
                            stmt = stmt.on_conflict_do_nothing(index_elements=['ticker', 'timestamp', 'interval'])
                            session.execute(stmt)
                        session.commit()
                        print(f"[{ticker}] Hydrated DB with {len(records)} new {interval} bars.")
                    except Exception as e:
                        session.rollback()
                        print(f"[{ticker}] DB Insert Failed: {e}")

        # Serve the full cached dataset — no start filter for daily/weekly so
        # the GUI can scroll back as far as history exists.
        with Session(self.db.engine) as session:
            where_clauses = [
                OHLCV.ticker == ticker,
                OHLCV.interval == interval,
            ]
            if max_lookback is not None:
                # Intraday: still filter to the clamped window
                where_clauses.append(OHLCV.timestamp >= start)
            stmt = select(OHLCV).where(*where_clauses).order_by(OHLCV.timestamp.asc())

            df_final = pd.read_sql(stmt, session.bind)
            if not df_final.empty:
                df_final.set_index('timestamp', inplace=True)

        return df_final[['open', 'high', 'low', 'close', 'volume']]