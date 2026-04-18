import os
from datetime import datetime
from typing import Optional, Tuple
import pandas as pd
from sqlalchemy import (
    create_engine, Column, String, Float, DateTime, Integer,
    Index, text, event, func,
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import NullPool

InsiderBase = declarative_base()


class InsiderTransaction(InsiderBase):
    __tablename__ = "insider_transactions"

    id               = Column(Integer, primary_key=True)
    ticker           = Column(String, index=True)
    filing_date      = Column(DateTime)
    transaction_date = Column(DateTime, index=True)
    transaction_type = Column(String)   # 'P' = open-market purchase
    price            = Column(Float)
    shares           = Column(Float)
    insider_name     = Column(String)
    insider_role     = Column(String)
    accession_number = Column(String, unique=True)

    __table_args__ = (
        Index("idx_insider_ticker_txdate", "ticker", "transaction_date"),
    )


class InsiderDatabase:
    """SQLite cache for SEC EDGAR Form 4 insider transactions."""

    def __init__(self, db_path: str = "data/insider.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_path}", poolclass=NullPool)
        event.listen(self.engine, "connect", self._set_pragma)
        InsiderBase.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    @staticmethod
    def _set_pragma(dbapi_conn, _record):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        cur.close()

    def get_cached_range(self, ticker: str) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Returns (min_txdate, max_txdate) for cached data, or (None, None)."""
        session = self.Session()
        try:
            result = session.query(
                func.min(InsiderTransaction.transaction_date),
                func.max(InsiderTransaction.transaction_date),
            ).filter(InsiderTransaction.ticker == ticker).first()
            return (result[0], result[1])
        finally:
            session.close()

    def save_transactions(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        with self.engine.begin() as conn:
            for row in df.to_dict("records"):
                conn.execute(
                    text("""
                        INSERT OR IGNORE INTO insider_transactions
                          (ticker, filing_date, transaction_date, transaction_type,
                           price, shares, insider_name, insider_role, accession_number)
                        VALUES
                          (:ticker, :filing_date, :transaction_date, :transaction_type,
                           :price, :shares, :insider_name, :insider_role, :accession_number)
                    """),
                    row,
                )

    def get_transactions(
        self, ticker: str, start: datetime, end: datetime
    ) -> pd.DataFrame:
        with self.engine.begin() as conn:
            rows = conn.execute(
                text("""
                    SELECT ticker, filing_date, transaction_date, transaction_type,
                           price, shares, insider_name, insider_role, accession_number
                    FROM insider_transactions
                    WHERE ticker = :ticker
                      AND transaction_date BETWEEN :start AND :end
                    ORDER BY transaction_date
                """),
                {"ticker": ticker, "start": start, "end": end},
            ).fetchall()

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows, columns=[
            "ticker", "filing_date", "transaction_date", "transaction_type",
            "price", "shares", "insider_name", "insider_role", "accession_number",
        ])
