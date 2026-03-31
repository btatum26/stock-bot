"""
ModelEngine — GUI-facing facade for the research engine.

Wraps strategy management, data access, and the three execution modes
(backtest, train, signal) behind a single object the GUI can hold onto.
All execution methods accept a `callbacks` dict so they can report
progress and cancellation back to a QThread without importing Qt.
"""

import os
import json
import zipfile
from datetime import datetime, timedelta
from typing import List

import pandas as pd

from engine.core.data_broker.data_broker import DataBroker
from engine.core.data_broker.database import Database
from engine.core.features.base import FEATURE_REGISTRY
from engine.core.features.features import load_features
from engine.core.workspace import WorkspaceManager
from engine.core.backtester import LocalBacktester
from engine.core.bundler import Bundler
from engine.core.metrics import Tearsheet
from engine.core.exceptions import StrategyError, ValidationError

_REQUIRED_MANIFEST_KEYS = {"features", "hyperparameters", "parameter_bounds"}

# Bars needed for indicator warm-up, keyed by interval
_WARMUP_WINDOW: dict = {
    "1wk": timedelta(weeks=300),
    "1d":  timedelta(days=420),
    "4h":  timedelta(days=150),
    "1h":  timedelta(days=50),
    "15m": timedelta(days=10),
}


class ModelEngine:
    """
    Single-import façade for the Model Engine backend.

    Bootstraps the engine once and exposes a clean interface for the GUI
    (or any caller) to drive strategy management, data access, and execution.
    """

    def __init__(self, workspace_dir: str, db_path: str):
        self.workspace_dir = os.path.normpath(workspace_dir)
        load_features()
        self._broker = DataBroker()
        self._broker.db = Database(db_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _strategy_dir(self, name: str) -> str:
        return os.path.join(self.workspace_dir, name)

    def _load_manifest(self, strategy_name: str) -> dict:
        path = os.path.join(self._strategy_dir(strategy_name), "manifest.json")
        if not os.path.exists(path):
            raise StrategyError(f"No manifest.json found for strategy '{strategy_name}'")
        with open(path, "r") as f:
            return json.load(f)

    @staticmethod
    def _parse_dt(value: str) -> datetime:
        return datetime.fromisoformat(value)

    # ------------------------------------------------------------------
    # Strategy & Workspace Management
    # ------------------------------------------------------------------

    def list_strategies(self) -> List[str]:
        if not os.path.isdir(self.workspace_dir):
            return []
        result = []
        for name in os.listdir(self.workspace_dir):
            candidate = os.path.join(self.workspace_dir, name)
            manifest = os.path.join(candidate, "manifest.json")
            if os.path.isdir(candidate) and os.path.exists(manifest):
                result.append(name)
        return sorted(result)

    def create_strategy(self, strategy_name: str) -> str:
        """Scaffold a new strategy workspace in workspace_dir.

        Creates the directory, writes a default manifest, and generates
        context.py and model.py via WorkspaceManager.

        Returns:
            str: The name of the newly created strategy.

        Raises:
            StrategyError: If the name is invalid or already exists.
        """
        if not strategy_name or not strategy_name.isidentifier():
            raise StrategyError(
                f"'{strategy_name}' is not a valid strategy name. "
                "Use letters, digits, and underscores only."
            )

        strat_dir = self._strategy_dir(strategy_name)
        if os.path.exists(strat_dir):
            raise StrategyError(f"Strategy '{strategy_name}' already exists.")

        os.makedirs(strat_dir)
        try:
            default_features = [
                {"id": "RSI",           "params": {"period": 14}},
                {"id": "BollingerBands","params": {"period": 20, "std_dev": 2.0}},
            ]
            default_hparams = {"stop_loss": 0.05, "take_profit": 0.10}
            default_bounds  = {"stop_loss": [0.01, 0.10]}

            wm = WorkspaceManager(strategy_dir=strat_dir)
            wm.sync(
                features=default_features,
                hparams=default_hparams,
                bounds=default_bounds,
            )
        except Exception as e:
            import shutil
            shutil.rmtree(strat_dir, ignore_errors=True)
            raise StrategyError(f"Failed to scaffold strategy '{strategy_name}': {e}") from e

        return strategy_name

    def get_strategy_config(self, strategy_name: str) -> dict:
        return self._load_manifest(strategy_name)

    def save_strategy_config(self, strategy_name: str, config: dict) -> None:
        strategy_dir = self._strategy_dir(strategy_name)
        if not os.path.isdir(strategy_dir):
            raise StrategyError(f"Strategy directory not found: {strategy_dir}")

        manifest_path = os.path.join(strategy_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(config, f, indent=4)

        wm = WorkspaceManager(strategy_dir)
        wm.sync(
            features=config.get("features", []),
            hparams=config.get("hyperparameters", {}),
            bounds=config.get("parameter_bounds", {}),
        )

    def export_strategy(self, strategy_name: str, export_path: str) -> str:
        strategy_dir = self._strategy_dir(strategy_name)
        return Bundler.export(strategy_dir, export_path)

    def import_strategy(self, file_path: str) -> str:
        if not file_path.endswith(".strat"):
            raise ValidationError("Import file must have a .strat extension")

        strategy_name = os.path.splitext(os.path.basename(file_path))[0]
        dest_dir = os.path.join(self.workspace_dir, strategy_name)

        with zipfile.ZipFile(file_path, "r") as zf:
            names = zf.namelist()
            if "manifest.json" not in names:
                raise ValidationError("Archive is missing manifest.json")

            raw_manifest = json.loads(zf.read("manifest.json"))
            missing = _REQUIRED_MANIFEST_KEYS - raw_manifest.keys()
            if missing:
                raise ValidationError(
                    f"Archive manifest is missing required keys: {missing}"
                )

            zf.extractall(dest_dir)

        return strategy_name

    # ------------------------------------------------------------------
    # Data & Feature Introspection
    # ------------------------------------------------------------------

    def get_historical_data(self, ticker: str, interval: str,
                            start: str = None, end: str = None) -> pd.DataFrame:
        return self._broker.get_data(
            ticker, interval,
            self._parse_dt(start) if start else None,
            self._parse_dt(end) if end else None,
        )

    def list_cached_tickers(self) -> List[str]:
        return self._broker.db.get_all_tickers()

    def get_available_features(self) -> List[dict]:
        result = []
        for feature_id, feature_cls in FEATURE_REGISTRY.items():
            instance = feature_cls()
            result.append(
                {
                    "id": feature_id,
                    "name": instance.name,
                    "description": instance.description,
                    "category": instance.category,
                    "params": instance.parameters,
                }
            )
        return result

    # ------------------------------------------------------------------
    # Execution (QThread targets)
    # ------------------------------------------------------------------

    def run_backtest(self, strategy_name: str, assets: List[str], timeframe: dict, callbacks: dict) -> dict:
        strategy_dir = self._strategy_dir(strategy_name)
        start = self._parse_dt(timeframe["start"])
        end = self._parse_dt(timeframe["end"])
        interval = timeframe.get("interval", "1d")

        datasets: dict = {}
        n = len(assets)
        for i, ticker in enumerate(assets):
            if callbacks["is_cancelled"]():
                return {"cancelled": True, "metrics": {}, "equity_curves": {}}

            callbacks["on_progress"](int(i / n * 40), f"Fetching {ticker}…")
            callbacks["on_log"](f"[Backtest] Fetching {ticker} ({interval})")

            df = self._broker.get_data(ticker, interval, start, end)
            if not df.empty:
                datasets[ticker] = df

        if not datasets:
            callbacks["on_progress"](100, "No data available.")
            return {"metrics": {}, "equity_curves": {}}

        callbacks["on_progress"](50, "Running vectorized backtest…")
        callbacks["on_log"]("[Backtest] Executing strategy batch")

        backtester = LocalBacktester(strategy_dir)
        batch_signals = backtester.run_batch(datasets)

        metrics_out: dict = {}
        equity_out: dict = {}
        signals_out: dict = {}
        n_done = len(batch_signals)

        for j, (ticker, signals) in enumerate(batch_signals.items()):
            if callbacks["is_cancelled"]():
                break

            callbacks["on_progress"](
                50 + int(j / max(n_done, 1) * 48), f"Scoring {ticker}…"
            )

            if signals.empty:
                metrics_out[ticker] = {"error": "No signals produced"}
                continue

            m = Tearsheet.calculate_metrics(datasets[ticker], signals)
            equity_curve: pd.Series = m.pop("equity_curve", pd.Series(dtype=float))

            if len(equity_curve) > 500:
                step = len(equity_curve) // 500
                equity_curve = equity_curve.iloc[::step]

            metrics_out[ticker] = m
            equity_out[ticker] = equity_curve.tolist()
            signals_out[ticker] = batch_signals[ticker]

        callbacks["on_progress"](100, "Backtest complete.")
        return {
            "metrics": metrics_out,
            "equity_curves": equity_out,
            "signals": signals_out,
        }

    def run_training(self, strategy_name: str, assets: List[str], timeframe: dict, callbacks: dict) -> dict:
        if callbacks["is_cancelled"]():
            return {"cancelled": True}

        from engine.core.controller import (
            ApplicationController,
            ExecutionMode,
            JobPayload,
            Timeframe,
        )

        callbacks["on_progress"](0, "Initializing training pipeline…")
        callbacks["on_log"](f"[Training] Strategy: {strategy_name}")

        payload = JobPayload(
            strategy=strategy_name,
            assets=assets,
            interval=timeframe.get("interval", "1d"),
            mode=ExecutionMode.TRAIN,
            timeframe=Timeframe(
                start=timeframe.get("start"),
                end=timeframe.get("end"),
            ),
        )

        controller = ApplicationController(strategies_dir=self.workspace_dir)
        result = controller.execute_job(payload)

        callbacks["on_progress"](100, "Training complete.")
        return result if isinstance(result, dict) else {"result": str(result)}

    def generate_signals(self, strategy_name: str, assets: List[str], callbacks: dict) -> dict:
        strategy_dir = self._strategy_dir(strategy_name)
        manifest = self._load_manifest(strategy_name)
        interval = manifest.get("interval", "1d")

        end = datetime.utcnow()
        start = end - _WARMUP_WINDOW.get(interval, timedelta(days=420))

        datasets: dict = {}
        n = len(assets)
        for i, ticker in enumerate(assets):
            if callbacks["is_cancelled"]():
                return {}

            callbacks["on_progress"](int(i / n * 50), f"Fetching {ticker}…")
            callbacks["on_log"](f"[Signal] Fetching {ticker}")

            df = self._broker.get_data(ticker, interval, start, end)
            if not df.empty:
                datasets[ticker] = df

        if not datasets:
            callbacks["on_progress"](100, "No data available.")
            return {}

        callbacks["on_progress"](60, "Evaluating model…")
        callbacks["on_log"]("[Signal] Running feature DAG and model")

        backtester = LocalBacktester(strategy_dir)
        batch_signals = backtester.run_batch(datasets)

        out: dict = {}
        for ticker, signals in batch_signals.items():
            if callbacks["is_cancelled"]():
                break
            if not signals.empty:
                out[ticker] = float(signals.iloc[-1])

        callbacks["on_progress"](100, "Signals ready.")
        return out
