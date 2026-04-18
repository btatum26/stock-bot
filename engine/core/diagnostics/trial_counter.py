"""Persistent trial counter for strategy diagnostics.

Every time a backtest or training job runs for a given strategy, the counter
is incremented. The total count is consumed by the Deflated Sharpe Ratio
calculation so that the expected-maximum-SR benchmark grows with each trial.
"""
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

_DB_PATH = Path(__file__).parents[3] / "data" / "diagnostics.db"


def _conn() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trial_counts (
            strategy_name  TEXT    PRIMARY KEY,
            backtest_count INTEGER DEFAULT 0,
            train_count    INTEGER DEFAULT 0,
            last_run_at    TEXT
        )
    """)
    conn.commit()
    return conn


def increment(strategy_name: str, mode: str) -> None:
    """Increment backtest or train counter. mode: 'backtest' or 'train'."""
    col = "backtest_count" if mode == "backtest" else "train_count"
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as conn:
        conn.execute(
            f"""
            INSERT INTO trial_counts (strategy_name, {col}, last_run_at)
            VALUES (?, 1, ?)
            ON CONFLICT(strategy_name) DO UPDATE SET
                {col} = {col} + 1,
                last_run_at = excluded.last_run_at
            """,
            (strategy_name, now),
        )


def get_total_trials(strategy_name: str) -> int:
    """Return backtest_count + train_count. Returns 1 if never recorded."""
    try:
        with _conn() as conn:
            row = conn.execute(
                "SELECT backtest_count + train_count "
                "FROM trial_counts WHERE strategy_name = ?",
                (strategy_name,),
            ).fetchone()
        return int(row[0]) if row else 1
    except Exception:
        return 1


def get_all() -> list:
    """Return all rows as a list of dicts."""
    try:
        with _conn() as conn:
            rows = conn.execute(
                "SELECT strategy_name, backtest_count, train_count, "
                "backtest_count + train_count AS total, last_run_at "
                "FROM trial_counts ORDER BY strategy_name"
            ).fetchall()
        return [
            {
                "strategy_name": r[0],
                "backtest_count": r[1],
                "train_count": r[2],
                "total_count": r[3],
                "last_run_at": r[4],
            }
            for r in rows
        ]
    except Exception:
        return []


def set_counts(strategy_name: str, backtest_count: int, train_count: int) -> None:
    """Directly set counts (for backfill or manual correction)."""
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as conn:
        conn.execute(
            """
            INSERT INTO trial_counts (strategy_name, backtest_count, train_count, last_run_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(strategy_name) DO UPDATE SET
                backtest_count = excluded.backtest_count,
                train_count    = excluded.train_count,
                last_run_at    = excluded.last_run_at
            """,
            (strategy_name, backtest_count, train_count, now),
        )
