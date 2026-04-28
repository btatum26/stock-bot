import os
import json
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from abc import ABC, abstractmethod
from enum import Enum
from pydantic import BaseModel, Field

from .data_broker.data_broker import DataBroker
from .workspace import WorkspaceManager
from .backtester import LocalBacktester
from .metrics import Tearsheet
from .trainer import LocalTrainer
from .logger import logger
from .exceptions import StrategyError, ValidationError
from .config import config


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    """Convert an optional date string to a datetime, or return None."""
    if value is None:
        return None
    return datetime.fromisoformat(value)

class ExecutionMode(str, Enum):
    TRAIN = "TRAIN"
    BACKTEST = "BACKTEST"
    SIGNAL_ONLY = "SIGNAL_ONLY"

class MultiAssetMode(str, Enum):
    BATCH = "BATCH"
    PORTFOLIO = "PORTFOLIO"

class Timeframe(BaseModel):
    start: Optional[str] = None
    end: Optional[str] = None

class JobPayload(BaseModel):
    strategy: str
    assets: List[str]
    interval: str
    timeframe: Timeframe = Field(default_factory=Timeframe)
    mode: ExecutionMode
    multi_asset_mode: MultiAssetMode = MultiAssetMode.BATCH

class SignalModel(ABC):
    """Interface for user-defined trading strategies.

    Rule-based (non-ML) strategies implement ``train()`` + ``generate_signals()``.

    ML strategies (``is_ml: true`` in manifest.json) implement
    ``build_labels()`` + ``fit_model()`` + ``generate_signals()``. The
    trainer calls ``build_labels`` once per ticker (so author code only
    ever sees a single asset), concatenates the resulting (X, y) across
    tickers, then calls ``fit_model`` once on the pooled matrix.

    Regime-aware strategies
    -----------------------
    Set ``"regime_aware": true`` and ``"regime_detector": "<name>"`` in
    manifest.json.  The backtester will then pass a ``regime_context``
    keyword argument to ``generate_signals``.  Declare it in your signature
    to receive it::

        def generate_signals(self, df, context, params, artifacts,
                             regime_context=None):
            if regime_context is not None:
                size = regime_context.size_multiplier
                proba = regime_context.proba     # pd.DataFrame (T x n_states)
                novelty = regime_context.novelty # pd.Series [0,1]

    Strategies that do not declare ``regime_context`` receive the original
    4-argument call and are unaffected — no manifest changes required.

    Cross-sectional strategies
    --------------------------
    Set ``"cross_sectional": true`` in manifest.json.  Instead of being
    called once per ticker with a single DataFrame, ``generate_signals``
    and ``train`` are each called **once** with the full universe::

        def train(self, data: Dict[str, pd.DataFrame], context, params) -> dict:
            # data maps ticker -> feature-computed, warmup-purged DataFrame
            return {}

        def generate_signals(self, data: Dict[str, pd.DataFrame], context,
                             params, artifacts) -> Dict[str, pd.Series]:
            # Compute cross-sectional z-scores, rank, etc.
            # Return one signal Series per ticker, values in [-1.0, 1.0].
            ...

    Each ticker's returned Series is validated and compressed independently.
    """

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame, context: Any, params: dict, artifacts: dict) -> pd.Series:
        """Execute signal generation logic.

        Returns a Pandas Series with values between -1.0 and 1.0.

        Regime-aware strategies may optionally declare a fifth keyword
        argument ``regime_context`` (see class docstring).
        """
        pass

    def train(self, df: pd.DataFrame, context: Any, params: dict) -> dict:
        """Rule-based training hook.

        Default no-op. Non-ML strategies override this to fit rules,
        thresholds, or cached intermediates. ML strategies ignore this
        and use ``build_labels`` + ``fit_model`` instead.
        """
        return {}

    def build_labels(self, df: pd.DataFrame, context: Any, params: dict) -> pd.Series:
        """ML label-construction hook. Called once per ticker.

        Given a single-asset DataFrame (features + OHLCV), return a
        Series aligned to ``df.index`` containing the target label per
        row. Use ``NaN`` for rows that cannot be labeled (e.g. the last
        ``lookforward`` bars where forward returns aren't available).

        The trainer drops NaN rows, concatenates across tickers, and
        hands the pooled matrix to ``fit_model``.
        """
        raise NotImplementedError(
            "ML strategies must override build_labels()"
        )

    def fit_model(self, X, y, params: dict) -> dict:
        """ML fit hook. Called once on the pooled (X, y) matrix.

        Args:
            X: 2D numpy array of shape ``(n_samples, n_features)`` with
                all tickers concatenated.
            y: 1D numpy array of labels, aligned to X.
            params: Strategy hyperparameters.

        Returns:
            Artifact dictionary (e.g. ``{"model": clf}``). The trainer
            will add ``feature_cols`` and ``system_scaler`` before
            inference.
        """
        raise NotImplementedError(
            "ML strategies must override fit_model()"
        )

class ApplicationController:
    """Orchestrates high-level system operations including data, workspace, and execution."""
    
    def __init__(self, strategies_dir: str = config.STRATEGIES_FOLDER):
        self.strategies_dir = strategies_dir
        self.broker = DataBroker()

    def execute_job(self, payload: Union[dict, JobPayload]) -> Any:
        """
        Primary entry point for job execution.
        Routes requests to backtesting, training, or signal generation pipelines.
        """
        try:
            if isinstance(payload, dict):
                payload = JobPayload(**payload)
        except Exception as e:
            logger.error(f"Invalid job payload: {e}", exc_info=True)
            raise ValidationError(f"Invalid job payload: {e}") from e
            
        strategy_name = payload.strategy
        assets = payload.assets
        interval = payload.interval
        mode = payload.mode
        multi_asset_mode = payload.multi_asset_mode

        if (
            not strategy_name
            or "/" in strategy_name
            or "\\" in strategy_name
            or strategy_name in {".", ".."}
            or os.path.basename(strategy_name) != strategy_name
        ):
            raise StrategyError(f"Invalid strategy name: {strategy_name!r}")

        strat_path = os.path.normpath(os.path.join(self.strategies_dir, strategy_name))
        if not os.path.exists(strat_path):
            logger.error(f"Strategy path not found: {strat_path}")
            raise StrategyError(f"Strategy {strategy_name} not found")

        # Route to appropriate handler
        if mode == ExecutionMode.BACKTEST:
            return self._handle_backtest(
                strat_path, 
                assets, 
                interval, 
                payload.timeframe.start, 
                payload.timeframe.end, 
                multi_asset_mode
            )
        
        elif mode == ExecutionMode.TRAIN:
            return self._handle_train(
                strat_path, 
                assets, 
                interval, 
                payload.timeframe, 
                payload
            )
        
        elif mode == ExecutionMode.SIGNAL_ONLY:
            return self._handle_signal_only(
                strat_path, 
                assets, 
                interval, 
                payload.timeframe.start, 
                payload.timeframe.end
            )
        
        else:
            raise ValidationError(f"Unsupported execution mode: {mode}")

    def _handle_backtest(self, strat_path: str, assets: List[str], interval: str,
                         start: Optional[str], end: Optional[str], multi_asset_mode: MultiAssetMode):
        """Executes the backtesting pipeline using batch processing."""
        if len(assets) > 1 and multi_asset_mode == MultiAssetMode.PORTFOLIO:
            raise NotImplementedError("API PORTFOLIO mode is not implemented; use the CLI/GUI portfolio path.")

        strategy_name = os.path.basename(strat_path)
        n_trials = 1
        try:
            from .diagnostics.trial_counter import increment, get_total_trials
            increment(strategy_name, "backtest")
            n_trials = get_total_trials(strategy_name)
        except Exception:
            pass

        # Fetch all data upfront
        datasets = {}
        for ticker in assets:
            logger.info(f"Fetching data for {ticker} ({interval})")
            df_raw = self.broker.get_data(ticker, interval, _parse_dt(start), _parse_dt(end))

            if df_raw.empty:
                logger.warning(f"No data available for {ticker} in requested range.")
                continue
            datasets[ticker] = df_raw

        all_metrics = {}
        if not datasets:
            return all_metrics

        # Run the backtester once for the entire batch
        try:
            backtester = LocalBacktester(strat_path)
            batch_signals = backtester.run_batch(datasets)

            # Calculate metrics for each completed run
            for ticker, signals in batch_signals.items():
                if signals.empty:
                    all_metrics[ticker] = {"error": "Execution failed during batch run"}
                    continue
                    
                metrics = Tearsheet.calculate_metrics(datasets[ticker], signals, n_trials=n_trials)
                # Store a copy without bulky time-series objects for the CLI summary
                scalar_metrics = {k: v for k, v in metrics.items()
                                  if k not in ("equity_curve", "portfolio", "bh_portfolio", "trade_log")}
                all_metrics[ticker] = scalar_metrics
                Tearsheet.print_summary(metrics)
                
        except Exception as e:
            logger.error(f"Batch backtest initialization failed: {e}", exc_info=True)
            for ticker in datasets.keys():
                all_metrics[ticker] = {"error": str(e)}

        return all_metrics

    def _handle_train(self, strat_path: str, assets: List[str], interval: str,
                      timeframe: Timeframe, payload: JobPayload):
        """Executes the optimization and training pipeline.

        Supports single- or multi-ticker training. For multi-ticker, all
        assets are fetched and passed to ``LocalTrainer`` as a dict; the
        trainer pools labels across tickers and fits one model on the
        stacked matrix.

        Two-phase process:
            **Phase A (optional):** Hyperparameter search via ``OptimizerCore``.
            **Phase B:** Data splitting, training, validation via ``LocalTrainer``.
        """
        if not assets:
            raise ValidationError("No assets provided for training")

        try:
            from .diagnostics.trial_counter import increment
            increment(os.path.basename(strat_path), "train")
        except Exception:
            pass

        # Fetch data for every requested ticker
        datasets: Dict[str, pd.DataFrame] = {}
        for ticker in assets:
            logger.info(f"Fetching data for {ticker} ({interval})")
            df = self.broker.get_data(
                ticker, interval,
                _parse_dt(timeframe.start), _parse_dt(timeframe.end),
            )
            if df.empty:
                logger.warning(f"No data available for {ticker}; skipping.")
                continue
            datasets[ticker] = df

        if not datasets:
            raise StrategyError("No data available for any requested ticker")

        logger.info(f"Training pipeline initialized for {list(datasets.keys())}")

        manifest_path = os.path.join(strat_path, "manifest.json")
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load manifest at {manifest_path}: {e}", exc_info=True)
            raise StrategyError(f"Could not load manifest for {strat_path}") from e

        optimal_params = manifest.get("hyperparameters", {})
        logger.info(f"Using manifest hyperparameters: {optimal_params}")

        # Train with proper data splitting and validation
        try:
            trainer = LocalTrainer(strat_path)
            # Single-ticker: unwrap to DataFrame for the legacy path.
            # Multi-ticker: pass the dict; trainer pools across tickers.
            if len(datasets) == 1:
                only_df = next(iter(datasets.values()))
                results = trainer.run(only_df, params=optimal_params)
            else:
                results = trainer.run(datasets, params=optimal_params)
            results["optimal_params"] = optimal_params
            return results
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            raise StrategyError(f"Training failed: {e}") from e

    def _handle_signal_only(self, strat_path: str, assets: Union[str, List[str]], interval: str, 
                            start: Optional[str], end: Optional[str]):
        """Executes the signal generation pipeline using batch processing."""
        # Normalize a single string asset into a list
        if isinstance(assets, str):
            assets = [assets]

        # Fetch all data upfront
        datasets = {}
        results = {}
        
        for ticker in assets:
            logger.info(f"Fetching data to generate signals for {ticker}")
            df_raw = self.broker.get_data(ticker, interval, _parse_dt(start), _parse_dt(end))

            if df_raw.empty:
                results[ticker] = {"error": "No data found"}
                continue
            datasets[ticker] = df_raw

        if not datasets:
            logger.warning("No datasets available for signal generation.")
            return results

        # Run the backtester once for the entire batch
        try:
            backtester = LocalBacktester(strat_path)
            batch_signals = backtester.run_batch(datasets)
            
            # Format the final output payloads
            for ticker, signals in batch_signals.items():
                if signals.empty:
                    results[ticker] = {"error": "Signal generation failed"}
                    continue
                    
                last_signal = float(signals.iloc[-1])
                timestamp = signals.index[-1].isoformat()

                results[ticker] = {
                    "signal": last_signal,
                    "timestamp": timestamp,
                    "asset": ticker,
                    "mode": "SIGNAL_ONLY"
                }
                
        except Exception as e:
            logger.error(f"Batch signal generation failed: {e}", exc_info=True)
            for ticker in datasets.keys():
                results[ticker] = {"error": str(e)}

        return results
