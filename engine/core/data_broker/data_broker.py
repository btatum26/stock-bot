import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Tuple
from sqlalchemy import select, func
from sqlalchemy.orm import Session

from .database import Database, OHLCV
from .fetcher import DataFetcher

class DataBroker:
    def __init__(self):
        self.db = Database()
        self.fetcher = DataFetcher()

    # Canonical interval names as stored in the DB (and as yfinance understands them)
    _INTERVAL_ALIASES = {'1w': '1wk'}

    def _normalize_interval(self, interval: str) -> str:
        return self._INTERVAL_ALIASES.get(interval, interval)

    def _get_padding(self, start_date: datetime, interval: str, periods: int = 200) -> datetime:
        """Calculates rough calendar padding to ensure indicator warm-up."""
        if interval == '1d':
            return start_date - timedelta(days=periods * 1.4)
        elif interval in ('1wk', '1w'):
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
    _EPOCH_START = datetime(1900, 1, 1)

    # How old the most-recent cached bar can be before we consider the DB stale.
    # Accounts for weekends, public holidays, and timezone offsets.
    _STALENESS_WINDOW: dict = {
        '1d':  timedelta(days=4),    # Fri close → Mon load is fine
        '1wk': timedelta(days=10),   # one full week + buffer
        '4h':  timedelta(days=2),
        '1h':  timedelta(days=2),
        '60m': timedelta(days=2),
    }

    def _needs_refresh(self, db_max_date, interval: str) -> bool:
        """True if the most-recent cached bar is old enough to need a forward-gap fetch."""
        if db_max_date is None:
            return True
        window = self._STALENESS_WINDOW.get(interval, timedelta(days=1))
        return (datetime.utcnow().date() - db_max_date) > window

    def _compute_fetch_range(
        self,
        start: datetime,
        end: datetime,
        db_min: Optional[datetime],
        db_max: Optional[datetime],
        db_max_date,
        interval: str,
        max_lookback,
    ) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        Returns (fetch_from, fetch_to) covering only data not already in the DB.
        Returns (None, None) when the cache is complete and fresh.

        Three cases:
          1. DB empty              → fetch full requested range
          2. Forward gap only      → fetch from last cached bar to end
          3. Backward gap only     → fetch from requested start to first cached bar
             (only applies to unlimited intervals; daily/weekly can have data
              added before the earliest cached bar)
          4. Both gaps             → fetch the full range in one shot
        """
        if not db_min or not db_max:
            # DB is empty — fetch everything
            return start, end

        needs_forward  = self._needs_refresh(db_max_date, interval)
        # Backward gap only relevant for unlimited (daily/weekly) intervals
        needs_backward = (max_lookback is None) and (start < db_min)

        if needs_forward and needs_backward:
            return start, end
        elif needs_forward:
            # Gap-fill: fetch from the last known bar so we don't miss partial bars
            return db_max, end
        elif needs_backward:
            return start, db_min
        else:
            return None, None   # cache is fresh and complete

    def _insert_dataframe(self, df_new: pd.DataFrame, ticker: str, interval: str):
        """Batch-inserts an OHLCV DataFrame into the DB, ignoring duplicates."""
        if df_new.empty:
            return
        df_new = df_new.copy()
        df_new['ticker']   = ticker
        df_new['interval'] = interval
        records = df_new.to_dict('records')

        # SQLite limit: 999 bound parameters per statement.
        # Each OHLCV row has 8 columns → max 124 rows per batch.
        _BATCH_SIZE = 999 // 8  # 124

        with Session(self.db.engine) as session:
            try:
                from sqlalchemy.dialects.sqlite import insert
                for i in range(0, len(records), _BATCH_SIZE):
                    batch = records[i:i + _BATCH_SIZE]
                    stmt  = insert(OHLCV).values(batch)
                    stmt  = stmt.on_conflict_do_nothing(
                        index_elements=['ticker', 'timestamp', 'interval'])
                    session.execute(stmt)
                session.commit()
                print(f"[{ticker}] Cached {len(records)} new {interval} bars.")
            except Exception as e:
                session.rollback()
                print(f"[{ticker}] DB insert failed: {e}")

    def get_data(self, ticker: str, interval: str, start: Optional[datetime] = None,
                 end: Optional[datetime] = None) -> pd.DataFrame:
        """
        Primary interface for the strategy engine and GUI.

        Fetches only the bars that are not already in the local DB:
          • If the DB is empty: fetches the full requested range.
          • If the DB has data but is stale: fetches only the forward gap
            (from the last cached bar to today).
          • If earlier history is missing (daily/weekly): fetches the backward
            gap (from the requested start to the first cached bar).
          • If the DB is fresh and complete: serves directly from the DB
            without touching yfinance at all.

        For daily/weekly intervals `start` is always overridden to _EPOCH_START
        so the full available history is stored; the GUI controls the view
        window independently via period/zoom.

        For intraday intervals `start` is clamped to Yahoo Finance's hard
        per-interval lookback limits.
        """
        interval = self._normalize_interval(interval)

        if end is None:
            end = datetime.utcnow()

        # Clamp intraday start dates to Yahoo Finance's hard per-interval limits
        max_lookback = self._YF_MAX_HISTORY.get(interval)
        if max_lookback is not None:
            if start is None:
                start = end - max_lookback
            else:
                earliest_allowed = end - max_lookback
                if start < earliest_allowed:
                    start = earliest_allowed
        else:
            # Daily / weekly: always store full history
            start = self._EPOCH_START

        # 15-Minute Rule: bypass DB (too short-lived to be worth caching)
        if interval == '15m':
            print(f"[{ticker}] 15m requested — bypassing DB.")
            df = self.fetcher.fetch_ohlcv(
                ticker, interval,
                start=start.strftime('%Y-%m-%d'),
                end=(end + timedelta(days=1)).strftime('%Y-%m-%d')
            )
            if not df.empty:
                df.set_index('timestamp', inplace=True)
            return df

        db_min, db_max = self._get_db_bounds(ticker, interval)
        db_max_date    = db_max.date() if db_max else None

        fetch_from, fetch_to = self._compute_fetch_range(
            start, end, db_min, db_max, db_max_date, interval, max_lookback
        )

        if fetch_from is not None:
            print(f"[{ticker}] Fetching {interval} gap: "
                  f"{fetch_from.strftime('%Y-%m-%d')} → {fetch_to.strftime('%Y-%m-%d')}")
            df_new = self.fetcher.fetch_ohlcv(
                ticker, interval,
                start=fetch_from.strftime('%Y-%m-%d'),
                end=(fetch_to + timedelta(days=1)).strftime('%Y-%m-%d')
            )
            self._insert_dataframe(df_new, ticker, interval)
        else:
            print(f"[{ticker}] {interval} cache is fresh — serving from DB.")

        # Serve from DB
        with Session(self.db.engine) as session:
            where_clauses = [
                OHLCV.ticker   == ticker,
                OHLCV.interval == interval,
            ]
            if max_lookback is not None:
                # Intraday: filter to the clamped window
                where_clauses.append(OHLCV.timestamp >= start)

            stmt     = select(OHLCV).where(*where_clauses).order_by(OHLCV.timestamp.asc())
            df_final = pd.read_sql(stmt, session.bind)
            if not df_final.empty:
                df_final.set_index('timestamp', inplace=True)
                df_final.index = pd.to_datetime(df_final.index)

        return df_final[['open', 'high', 'low', 'close', 'volume']]
