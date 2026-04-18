import logging
from typing import Dict, Any, List
import pandas as pd
from ..base import Feature, FeatureResult, OutputSchema, OutputType, Pane, register_feature

logger = logging.getLogger(__name__)

_db_instance  = None
_fetch_fn     = None


def _get_resources():
    global _db_instance, _fetch_fn
    if _db_instance is None:
        from ...data_broker.insider_db import InsiderDatabase
        from ...data_broker.insider_fetcher import fetch_insider_transactions
        _db_instance = InsiderDatabase()
        _fetch_fn    = fetch_insider_transactions
    return _db_instance, _fetch_fn


@register_feature("InsiderFlow")
class InsiderFlow(Feature):
    """Rolling officer/director open-market purchase cluster signal (SEC EDGAR Form 4).

    Data is fetched once and cached in data/insider.db. Subsequent backtests on the
    same ticker are served from the local cache.

    Parameters
    ----------
    ticker : str  (required)
        The equity ticker to analyse.
    window : int
        Rolling window in calendar days for the purchase count (default 14).
    min_cluster : int
        Minimum purchase count within the window to emit a cluster marker (default 3).

    Outputs
    -------
    purchase_count : rolling count of insider purchases in the window
    cluster        : sparse marker — non-NaN only where count >= min_cluster
    """

    @property
    def name(self) -> str:
        return "Insider Flow"

    @property
    def description(self) -> str:
        return (
            "Rolling count of insider open-market purchases (officers/directors). "
            "Cluster marker at 3+ purchases in the window."
        )

    @property
    def category(self) -> str:
        return "Alternative"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "ticker":      "",
            "window":      14,
            "min_cluster": 3,
        }

    @property
    def output_schema(self) -> List[OutputSchema]:
        return [
            OutputSchema(name="purchase_count", output_type=OutputType.LINE,   pane=Pane.NEW),
            OutputSchema(name="cluster",        output_type=OutputType.MARKER, pane=Pane.NEW),
        ]

    def compute(self, df: pd.DataFrame, params: Dict[str, Any], cache: Any = None) -> FeatureResult:
        ticker      = params.get("ticker", "")
        window      = int(params.get("window", 14))
        min_cluster = int(params.get("min_cluster", 3))

        col_count   = self.generate_column_name("InsiderFlow", params, "purchase_count")
        col_cluster = self.generate_column_name("InsiderFlow", params, "cluster")
        nan         = pd.Series(float("nan"), index=df.index)

        if not ticker:
            logger.warning("InsiderFlow: 'ticker' param is required")
            return FeatureResult(data={col_count: nan, col_cluster: nan})

        start_str = df.index.min().strftime("%Y-%m-%d")
        end_str   = df.index.max().strftime("%Y-%m-%d")

        db, fetch = _get_resources()

        try:
            db_min, db_max = db.get_cached_range(ticker)
            needs_fetch = (
                db_min is None
                or db_min.strftime("%Y-%m-%d") > start_str
                or db_max.strftime("%Y-%m-%d") < end_str
            )
            if needs_fetch:
                fresh = fetch(ticker, start_str, end_str)
                if not fresh.empty:
                    db.save_transactions(fresh)

            txns = db.get_transactions(ticker, df.index.min(), df.index.max())
        except Exception as e:
            logger.warning(f"InsiderFlow data retrieval failed for {ticker}: {e}")
            return FeatureResult(data={col_count: nan, col_cluster: nan})

        zeros = pd.Series(0.0, index=df.index)

        if txns.empty:
            return FeatureResult(data={col_count: zeros, col_cluster: nan})

        purchases = txns[txns["transaction_type"] == "P"].copy()
        purchases["transaction_date"] = pd.to_datetime(purchases["transaction_date"])

        daily = purchases.groupby("transaction_date").size().reindex(df.index, fill_value=0)

        rolling_count = daily.rolling(window=window, min_periods=1).sum()
        cluster       = rolling_count.where(rolling_count >= min_cluster)

        return FeatureResult(data={
            col_count:   rolling_count,
            col_cluster: cluster,
        })
