"""
ModelEngine — GUI-facing facade for the research engine.

Wraps strategy management, data access, and the three execution modes
(backtest, train, signal) behind a single object the GUI can hold onto.
All execution methods accept a `callbacks` dict so they can report
progress and cancellation back to a QThread without importing Qt.
"""

import os
import json
import logging
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


class _CallbackLogHandler(logging.Handler):
    """Forwards engine log records to a GUI callback.

    Installed on the 'model-engine' logger while a job is running so the
    training panel's text box mirrors what's printed to the console.
    """

    def __init__(self, callback):
        super().__init__()
        self._callback = callback

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._callback(self.format(record))
        except Exception:
            pass

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

    def save_training_config(self, strategy_name: str, training_config: dict) -> None:
        """Update only the 'training' section of a strategy's manifest.

        Does not touch features, hyperparameters, or regenerate context.py.
        """
        manifest = self._load_manifest(strategy_name)
        manifest["training"] = training_config
        manifest_path = os.path.join(self._strategy_dir(strategy_name), "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=4)

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

    def run_backtest(self, strategy_name: str, assets: List[str], timeframe: dict, callbacks: dict,
                     starting_capital: float = 10_000.0) -> dict:
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
        portfolio_out: dict = {}
        bh_portfolio_out: dict = {}
        trade_log_out: dict = {}
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

            m = Tearsheet.calculate_metrics(
                datasets[ticker], signals, starting_capital=starting_capital
            )

            # Pop time-series objects before storing scalar metrics
            equity_curve: pd.Series = m.pop("equity_curve", pd.Series(dtype=float))
            portfolio: pd.Series = m.pop("portfolio", pd.Series(dtype=float))
            bh_portfolio: pd.Series = m.pop("bh_portfolio", pd.Series(dtype=float))
            trade_log: pd.DataFrame = m.pop("trade_log", pd.DataFrame())

            # Downsample long series so JSON payloads stay manageable
            def _downsample(s: pd.Series, max_points: int = 500) -> list:
                if len(s) > max_points:
                    s = s.iloc[:: len(s) // max_points]
                # Convert DatetimeIndex to ISO strings for JSON serialisation
                return [
                    {"t": str(idx), "v": round(float(v), 6)}
                    for idx, v in s.items()
                ]

            metrics_out[ticker] = m
            equity_out[ticker] = _downsample(equity_curve)
            portfolio_out[ticker] = _downsample(portfolio)
            bh_portfolio_out[ticker] = _downsample(bh_portfolio)
            signals_out[ticker] = batch_signals[ticker]

            # Trade log: convert Timestamps to strings for JSON
            if not trade_log.empty:
                tl = trade_log.copy()
                for col in ('entry_date', 'exit_date'):
                    if col in tl.columns:
                        tl[col] = tl[col].astype(str)
                trade_log_out[ticker] = tl.to_dict('records')
            else:
                trade_log_out[ticker] = []

        callbacks["on_progress"](100, "Backtest complete.")
        return {
            "metrics": metrics_out,
            "equity_curves": equity_out,
            "portfolios": portfolio_out,
            "bh_portfolios": bh_portfolio_out,
            "trade_logs": trade_log_out,
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

        # Mirror 'model-engine' logger records into the GUI log panel for
        # the duration of this training run. Child loggers propagate up,
        # so this catches trainer, controller, optimizer, etc.
        engine_logger = logging.getLogger("model-engine")
        gui_handler = _CallbackLogHandler(callbacks["on_log"])
        gui_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        engine_logger.addHandler(gui_handler)

        try:
            controller = ApplicationController(strategies_dir=self.workspace_dir)
            result = controller.execute_job(payload)
        finally:
            engine_logger.removeHandler(gui_handler)

        callbacks["on_progress"](100, "Training complete.")
        return result if isinstance(result, dict) else {"result": str(result)}

    def write_context_py(self, strategy_name: str, features_list: list, hparams: dict) -> None:
        """Regenerate context.py for a strategy from its current features and hyperparameters.

        Attribute naming rules:
          - Single unnamed output (e.g. RSI):  attr = col_name  (e.g. RSI_14)
          - Named output (e.g. fractal cols):  attr = OUTPUT_NAME_UPPER
          - Collision (two features share an output name): prefix with FEATUREID_
        """
        from engine.core.features.base import FEATURE_REGISTRY, OutputType

        DATA_TYPES = (OutputType.LINE, OutputType.HISTOGRAM, OutputType.MARKER)
        strategy_dir = self._strategy_dir(strategy_name)

        lines = [
            "# AUTO-GENERATED. Do not edit — updated by the GUI when features change.",
            "from dataclasses import dataclass, field",
            "",
            "",
            "@dataclass(frozen=True)",
            "class FeaturesContext:",
            '    """Typed mapping from strategy features to DataFrame column names."""',
        ]

        seen_attrs: set = set()
        for entry in features_list:
            fid    = entry["id"]
            params = entry.get("params", {})
            if fid not in FEATURE_REGISTRY:
                continue
            feat         = FEATURE_REGISTRY[fid]()
            data_outputs = [s for s in feat.output_schema if s.output_type in DATA_TYPES]
            source_keys  = set(getattr(feat, 'source_param_keys', []))
            name_params  = {k: v for k, v in params.items() if k not in source_keys}

            for schema in data_outputs:
                col_name = feat.generate_column_name(fid, name_params, schema.name)
                if schema.name:
                    attr = schema.name.upper().replace(" ", "_").replace("-", "_")
                else:
                    attr = col_name
                if attr in seen_attrs:
                    attr = f"{fid.upper().replace(' ', '_')}_{attr}"
                seen_attrs.add(attr)
                lines.append(f"    {attr}: str = '{col_name}'")

        lines += [
            "",
            "",
            "@dataclass(frozen=True)",
            "class ParamsContext:",
            '    """Typed strategy hyperparameters."""',
        ]
        for k, v in hparams.items():
            hint = "float" if isinstance(v, float) else "int" if isinstance(v, int) else "str"
            lines.append(f"    {k}: {hint} = {repr(v)}")

        lines += [
            "",
            "",
            "@dataclass(frozen=True)",
            "class Context:",
            '    """Master context object."""',
            "    features: FeaturesContext = field(default_factory=FeaturesContext)",
            "    params:   ParamsContext   = field(default_factory=ParamsContext)",
            "",
        ]

        with open(os.path.join(strategy_dir, "context.py"), "w") as f:
            f.write("\n".join(lines))

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
