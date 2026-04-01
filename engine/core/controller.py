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
from .optimization.optimizer_core import OptimizerCore
from .logger import logger
from .exceptions import StrategyError, ValidationError
from .config import config

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
    """Interface for user-defined trading strategies."""
    
    @abstractmethod
    def train(self, df: pd.DataFrame, context: Any, params: dict) -> dict:
        """
        Execute the training logic for the strategy.
        Returns a dictionary of generated artifacts (e.g. weights, thresholds).
        """
        pass

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame, context: Any, params: dict, artifacts: dict) -> pd.Series:
        """
        Execute signal generation logic.
        Returns a Pandas Series with values between -1.0 and 1.0.
        """
        pass

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
            raise ValidationError(f"Invalid job payload: {e}")
            
        strategy_name = payload.strategy
        assets = payload.assets
        interval = payload.interval
        mode = payload.mode
        multi_asset_mode = payload.multi_asset_mode

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
            raise NotImplementedError("PORTFOLIO mode is not yet supported.")

        # Fetch all data upfront
        datasets = {}
        for ticker in assets:
            logger.info(f"Fetching data for {ticker} ({interval})")
            df_raw = self.broker.get_data(ticker, interval, start, end)
            
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
                    
                metrics = Tearsheet.calculate_metrics(datasets[ticker], signals)
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
        """Executes the optimization and training pipeline."""
        ticker = assets[0]
        dataset_ref = f"{ticker}_{interval}"
        
        logger.info(f"Initializing optimization for {ticker}")
        
        # Ensure data is cached before optimization starts
        self.broker.get_data(ticker, interval, timeframe.start, timeframe.end)

        manifest_path = os.path.join(strat_path, "manifest.json")
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load manifest at {manifest_path}: {e}")
            raise StrategyError(f"Could not load manifest for {strat_path}")

        try:
            optimizer = OptimizerCore(
                strategy_path=strat_path,
                dataset_ref=dataset_ref,
                manifest=manifest,
                ticker=ticker,
                interval=interval,
                start=timeframe.start,
                end=timeframe.end
            )
            return optimizer.run()
        except Exception as e:
            logger.error(f"Optimization failed: {e}", exc_info=True)
            raise StrategyError(f"Optimization failed: {e}")

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
            df_raw = self.broker.get_data(ticker, interval, start, end)
            
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