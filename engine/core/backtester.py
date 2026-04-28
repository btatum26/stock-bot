import inspect
import json
import importlib.util
import sys
import os
import pandas as pd
import itertools
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from .features.features import compute_all_features
from .ml_bridge.orchestrator import MLBridge
from .ml_bridge.artifact_manager import ArtifactManager
from .logger import logger
from .exceptions import StrategyError

# ---------------------------------------------------------------------------
# Tearsheet — formerly engine/core/metrics.py
# ---------------------------------------------------------------------------

class Tearsheet:
    """
    Translates raw signals into trading reality.

    Provides two complementary views of strategy performance:
      - calculate_metrics(): continuous fractional-allocation model used for
        optimization fitness scoring (smooth gradients, Sharpe as objective).
      - Discrete simulation embedded within calculate_metrics(): thresholds
        signals to real -1/0/+1 positions, tracks a dollar portfolio, and
        builds a per-trade log so callers can visualize exactly where money
        went.
    """

    @staticmethod
    def calculate_metrics(
        df: pd.DataFrame,
        signals: pd.Series,
        friction: float = 0.001,
        starting_capital: float = 10_000.0,
        entry_threshold: float = 0.2,
        n_trials: int = 1,
    ) -> Dict[str, Any]:
        """
        Calculates performance metrics using a T+1 execution model.

        Signals generated at bar [T] are executed at bar [T+1] open.
        Returns are realised from [T+1] open to [T+2] open.

        Args:
            df: OHLCV DataFrame with lowercase column names and a DatetimeIndex.
            signals: Conviction signals in [-1.0, 1.0].
            friction: Round-trip transaction cost per trade (default 10 bps).
            starting_capital: Dollar amount for the discrete portfolio simulation.
            entry_threshold: Minimum absolute signal magnitude to open a position
                in the discrete simulation (default 0.2 = 20% conviction).

        Returns:
            Dict containing scalar metrics plus three time-series objects:
              - ``equity_curve`` (pd.Series): normalised growth index from 1.0
                (used by the optimiser as a smooth fitness surface).
              - ``portfolio`` (pd.Series): dollar value of the discrete portfolio.
              - ``bh_portfolio`` (pd.Series): buy-and-hold dollar benchmark.
              - ``trade_log`` (pd.DataFrame): one row per discrete round trip.
        """
        # ------------------------------------------------------------------
        # 1. Continuous signal model (optimiser fitness)
        # ------------------------------------------------------------------
        # Return from T+1 open to T+2 open, aligned back to bar T.
        # Reindex to signals.index so all downstream operations share the same index
        # (df may be the full raw frame while signals come from the warmup-purged slice).
        returns = df['open'].pct_change().shift(-2).reindex(signals.index)
        # Clip raw bar returns to [-0.5, 0.5] to prevent extreme gaps (splits,
        # halts, earnings shocks) from driving strategy_returns below -1.0,
        # which would flip the equity curve negative and break CAGR math.
        returns = returns.clip(lower=-0.5, upper=0.5)
        strategy_returns = signals * returns

        # Friction fires on every change in signal magnitude (continuous model).
        trades_mask = signals.diff().fillna(0).abs()
        strategy_returns -= trades_mask * friction

        equity_curve = (1 + strategy_returns.fillna(0)).cumprod()

        total_return = (equity_curve.iloc[-1] - 1) * 100
        total_trades_continuous = int((trades_mask > 0).sum())

        # Use median bar duration x bar count so CPCV non-contiguous training
        # folds don't inflate `days` by spanning the full calendar range of
        # the dataset (which would include gaps belonging to validation groups).
        # Use pandas Timedelta arithmetic instead of astype(int64), since
        # pandas 2.x DatetimeIndex can have non-ns resolution and int64 casts
        # would return seconds/micros and collapse `days` to zero.
        if len(df.index) > 1:
            diffs_sec = (
                df.index.to_series().diff().dropna().dt.total_seconds()
            )
            median_bar_sec = float(diffs_sec.median()) if not diffs_sec.empty else 0.0
            days = max(int(median_bar_sec * len(equity_curve) / 86400), 1)
        else:
            days = 1
        end_equity = float(equity_curve.iloc[-1])
        if end_equity > 0 and days > 0:
            log_eq = np.log(end_equity) if np.isfinite(end_equity) else np.inf
            exponent = log_eq * (365.25 / days)
            # Cap the exponent to prevent absurd CAGR values from short
            # CPCV validation windows where small gains annualize to millions.
            exponent = max(min(exponent, 10.0), -10.0)
            cagr = (np.exp(exponent) - 1) * 100 if np.isfinite(exponent) else float('inf')
        else:
            cagr = -100.0

        trade_returns = strategy_returns[trades_mask > 0]
        win_rate = (trade_returns > 0).mean() * 100 if not trade_returns.empty else 0.0

        gains = strategy_returns[strategy_returns > 0].sum()
        losses = abs(strategy_returns[strategy_returns < 0].sum())
        profit_factor = gains / losses if losses > 0 else float('inf')

        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100  # negative number, e.g. -25.3

        volatility = strategy_returns.std() * np.sqrt(252)
        sharpe = (strategy_returns.mean() * 252) / volatility if volatility > 0 else 0.0

        dsr_val = float("nan")
        try:
            from .diagnostics.dsr import compute_dsr
            dsr_val = compute_dsr(strategy_returns.dropna(), n_trials)
        except Exception:
            pass

        downside = strategy_returns[strategy_returns < 0]
        downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else 0.0
        sortino = (strategy_returns.mean() * 252) / downside_vol if downside_vol > 0 else 0.0

        calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0.0

        avg_win = trade_returns[trade_returns > 0].mean() * 100 if (trade_returns > 0).any() else 0.0
        avg_loss = trade_returns[trade_returns < 0].mean() * 100 if (trade_returns < 0).any() else 0.0
        win_rate_decimal = win_rate / 100
        expectancy = (win_rate_decimal * avg_win) + ((1 - win_rate_decimal) * avg_loss)

        # ------------------------------------------------------------------
        # 2. Discrete position simulation (dollar portfolio)
        # ------------------------------------------------------------------
        position = pd.Series(0.0, index=signals.index)
        position[signals >= entry_threshold] = 1.0
        position[signals <= -entry_threshold] = -1.0

        discrete_friction = position.diff().abs().fillna(0) * friction
        discrete_returns = (position * returns) - discrete_friction
        portfolio = pd.Series(
            starting_capital * (1 + discrete_returns.fillna(0)).cumprod(),
            index=signals.index,
        )

        bh_portfolio = pd.Series(
            starting_capital * (1 + returns.fillna(0)).cumprod(),
            index=signals.index,
        )

        # ------------------------------------------------------------------
        # 3. Trade log (one row per discrete round trip)
        # ------------------------------------------------------------------
        trade_log = Tearsheet._build_trade_log(position, df)

        total_trades_discrete = len(trade_log)
        discrete_win_rate = (
            (trade_log['return_pct'] > 0).mean() * 100
            if not trade_log.empty else 0.0
        )

        return {
            # Growth
            "Total Return (%)":    round(total_return, 2),
            "CAGR (%)":            round(cagr, 2),
            # Risk / quality
            "Sharpe Ratio":        round(sharpe, 2),
            "Deflated Sharpe Ratio": round(dsr_val, 4) if np.isfinite(dsr_val) else float("nan"),
            "Sortino Ratio":       round(sortino, 2),
            "Calmar Ratio":        round(calmar, 2),
            "Max Drawdown (%)":    round(max_drawdown, 2),
            # Trade mechanics
            "Win Rate (%)":        round(win_rate, 2),
            "Avg Win (%)":         round(avg_win, 4),
            "Avg Loss (%)":        round(avg_loss, 4),
            "Expectancy (%)":      round(expectancy, 4),
            "Profit Factor":       round(profit_factor, 2),
            "Total Trades":        total_trades_continuous,
            # Discrete simulation results
            "Discrete Trades":     total_trades_discrete,
            "Discrete Win Rate (%)": round(discrete_win_rate, 2),
            # Time-series objects (popped by callers before serialising)
            "equity_curve":        equity_curve,
            "portfolio":           portfolio,
            "bh_portfolio":        bh_portfolio,
            "trade_log":           trade_log,
        }

    @staticmethod
    def _build_trade_log(position: pd.Series, df: pd.DataFrame) -> pd.DataFrame:
        """
        Builds a trade log from a discrete position series.

        Each row represents one round-trip trade: from the bar where a non-zero
        position first appears to the bar where it ends.  Entry and exit prices
        use bar opens (T+1 execution model).

        Args:
            position: Discrete positions series (-1.0, 0.0, 1.0).
            df: OHLCV DataFrame aligned to the same index as position.

        Returns:
            DataFrame with columns: entry_date, exit_date, direction,
            entry_price, exit_price, return_pct, bars_held.
        """
        empty = pd.DataFrame(columns=[
            'entry_date', 'exit_date', 'direction',
            'entry_price', 'exit_price', 'return_pct', 'bars_held',
        ])

        if position.empty or (position == 0).all():
            return empty

        opens = df['open']
        closes = df['close']

        # Every bar where the position value changes (entry, exit, or flip).
        pos_change = position.diff().fillna(position.iloc[0]) != 0
        # Entry bars: position changes AND the new value is non-zero.
        entry_bars = position.index[pos_change & (position != 0)].tolist()

        if not entry_bars:
            return empty

        trades = []
        for entry_bar in entry_bars:
            direction = int(position.loc[entry_bar])

            # Subsequent bars where the position changes again (exit or flip).
            subsequent = position.index[
                (position.index > entry_bar) & (position.diff().fillna(0) != 0)
            ]
            exit_signal_bar = subsequent[0] if len(subsequent) > 0 else None

            # Entry price = open of the bar after the signal bar (T+1 execution).
            entry_loc = opens.index.get_loc(entry_bar)
            if entry_loc + 1 >= len(opens):
                continue  # signal fires on the last bar — no room to enter
            actual_entry_date = opens.index[entry_loc + 1]
            entry_price = float(opens.iloc[entry_loc + 1])

            # Exit price = open of the bar after the exit-signal bar, or last close.
            if exit_signal_bar is not None:
                exit_loc = opens.index.get_loc(exit_signal_bar)
                if exit_loc + 1 < len(opens):
                    actual_exit_date = opens.index[exit_loc + 1]
                    exit_price = float(opens.iloc[exit_loc + 1])
                else:
                    actual_exit_date = opens.index[-1]
                    exit_price = float(closes.iloc[-1])
            else:
                actual_exit_date = opens.index[-1]
                exit_price = float(closes.iloc[-1])

            if entry_price <= 0:
                continue

            return_pct = (exit_price - entry_price) / entry_price * direction * 100

            entry_iloc = opens.index.get_loc(actual_entry_date)
            exit_iloc = opens.index.get_loc(actual_exit_date)
            bars_held = max(1, exit_iloc - entry_iloc)

            trades.append({
                'entry_date':   actual_entry_date,
                'exit_date':    actual_exit_date,
                'direction':    'LONG' if direction == 1 else 'SHORT',
                'entry_price':  round(entry_price, 4),
                'exit_price':   round(exit_price, 4),
                'return_pct':   round(return_pct, 4),
                'bars_held':    bars_held,
            })

        return pd.DataFrame(trades) if trades else empty

    @staticmethod
    def print_summary(metrics: Dict[str, Any]):
        """Logs scalar metrics in a compact report."""
        skip = {"equity_curve", "portfolio", "bh_portfolio", "trade_log"}
        lines = ["=" * 40, " " * 10 + "STRATEGY PERFORMANCE", "=" * 40]
        for key, value in metrics.items():
            if key in skip:
                continue
            lines.append(f"{key:<28}: {value}")
        lines.append("=" * 40)
        logger.info("\n%s", "\n".join(lines))


# ---------------------------------------------------------------------------
# LocalBacktester
# ---------------------------------------------------------------------------

class LocalBacktester:
    """
    Handles the local execution and testing of user-defined strategy models.

    This class is responsible for loading the strategy manifest, dynamically
    importing the user's Python scripts, computing Phase 2 features, purging
    warmup NaNs, and executing the strategy's signal generation logic.
    """

    def __init__(self, strategy_dir: str):
        """
        Initializes the backtester for a specific strategy directory.

        Args:
            strategy_dir (str): The path to the strategy folder containing the
                `manifest.json`, `model.py`, and `context.py` files.

        Raises:
            StrategyError: If the manifest file cannot be found or is invalid JSON.
        """
        self.strategy_dir = os.path.normpath(strategy_dir)
        self.manifest_path = os.path.join(self.strategy_dir, "manifest.json")
        try:
            with open(self.manifest_path, 'r') as f:
                self.manifest = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load manifest at {self.manifest_path}: {e}", exc_info=True)
            raise StrategyError(f"Missing or invalid manifest in {self.strategy_dir}")

    def _load_user_model_and_context(self) -> Tuple[type, Optional[type]]:
        """
        Dynamically imports the user-defined model and context classes.

        Returns:
            Tuple[type, Optional[type]]:
                - model_class (type): The uninstantiated user SignalModel subclass.
                - context_class (Optional[type]): The uninstantiated user Context class, if it exists.

        Raises:
            StrategyError: If `model.py` is missing or fails to import safely.
        """
        model_path = os.path.join(self.strategy_dir, "model.py")
        context_module_path = os.path.join(self.strategy_dir, "context.py")

        if not os.path.exists(model_path):
            raise StrategyError(f"model.py not found in {self.strategy_dir}")

        try:
            # Import context first
            context_module_name = f"user_context_{os.path.basename(self.strategy_dir)}"
            spec_ctx = importlib.util.spec_from_file_location(context_module_name, context_module_path)
            context_module = importlib.util.module_from_spec(spec_ctx)
            if spec_ctx and spec_ctx.loader:
                spec_ctx.loader.exec_module(context_module)
            context_class = getattr(context_module, 'Context', None)

            # Import model
            model_module_name = f"user_strat_{os.path.basename(self.strategy_dir)}"
            spec = importlib.util.spec_from_file_location(model_module_name, model_path)
            module = importlib.util.module_from_spec(spec)

            # Allow internal imports within the strategy directory
            if self.strategy_dir not in sys.path:
                sys.path.insert(0, self.strategy_dir)

            try:
                sys.modules[model_module_name] = module
                if spec and spec.loader:
                    spec.loader.exec_module(module)
            finally:
                if self.strategy_dir in sys.path:
                    sys.path.remove(self.strategy_dir)

            from .controller import SignalModel
            for obj_name in dir(module):
                obj = getattr(module, obj_name)
                if isinstance(obj, type) and issubclass(obj, SignalModel) and obj is not SignalModel:
                    return obj, context_class

            raise StrategyError(f"No valid SignalModel subclass found in {model_path}")
        except Exception as e:
            logger.error(f"Failed to load strategy components: {e}", exc_info=True)
            raise StrategyError(f"Strategy initialization failed: {e}")

    def _audit_nans(self, df: pd.DataFrame, feature_ids: List[str]):
        """
        Scans for NaN values in computed features and logs warnings.

        Note: Since the l_max purge handles rolling indicator warmup periods,
        any NaNs caught by this method indicate flawed data or broken indicators.

        Args:
            df (pd.DataFrame): The pre-processed dataset.
            feature_ids (List[str]): List of base feature names to audit.
        """
        for fid in feature_ids:
            cols = [c for c in df.columns if c.startswith(fid)]
            for col in cols:
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    logger.warning(f"Feature '{col}' has {nan_count} unexpected NaN values after l_max purge.")

    @staticmethod
    def _call_generate_signals_cross_sectional(model, data, context, hyperparams, artifacts):
        """Call generate_signals with the full universe dict (cross_sectional: true path)."""
        return model.generate_signals(data, context, hyperparams, artifacts)

    @staticmethod
    def _call_generate_signals(model, df, context, hyperparams, artifacts, regime_context=None):
        """Call model.generate_signals, passing regime_context only when the model accepts it.

        Strategies that were written before the regime system simply don't declare
        the parameter and will receive the original 4-argument call.  Regime-aware
        strategies declare ``regime_context`` as an optional keyword argument and
        receive the full context object.
        """
        if regime_context is not None:
            sig = inspect.signature(model.generate_signals)
            if "regime_context" in sig.parameters:
                return model.generate_signals(
                    df, context, hyperparams, artifacts, regime_context=regime_context
                )
        return model.generate_signals(df, context, hyperparams, artifacts)

    def _build_regime_context(self, df: pd.DataFrame):
        """Build a RegimeContext for df when the manifest requests it.

        Returns None if regime_aware is False or if regime detection fails.
        """
        if not self.manifest.get("regime_aware", False):
            return None
        try:
            from .regime.orchestrator import RegimeOrchestrator
            detector_name = self.manifest.get("regime_detector", "vix_adx")
            return RegimeOrchestrator().build_context(df, detector_name)
        except Exception as e:
            logger.warning(f"Regime detection failed, continuing without: {e}")
            return None

    def run(
        self,
        raw_data: pd.DataFrame,
        params: Optional[Dict[str, Any]] = None,
        artifacts: Optional[Dict[str, Any]] = None,
    ) -> pd.Series:
        """Executes a single vectorized backtest pass.

        When ``artifacts`` are provided (e.g. from a prior TRAIN run), the
        backtester skips ``model.train()`` and applies the saved scaler via
        ``MLBridge`` before calling ``generate_signals()`` directly. This is
        the intended path for SIGNAL_ONLY mode with trained ML strategies.

        When ``artifacts`` is ``None``, the backtester trains inline (calling
        ``model.train()`` on the same data used for signal generation). This
        is the standard BACKTEST path.

        Args:
            raw_data: The raw OHLCV market data with a ``DatetimeIndex``.
            params: Strategy hyperparameters. Defaults to the values in
                ``manifest.json``.
            artifacts: Pre-trained artifacts dictionary. When provided,
                ``model.train()`` is skipped and the saved scaler (if any)
                is applied via ``MLBridge``. When ``None``, loads persisted
                artifacts from disk for ML strategies, falling back to
                inline training if none exist.

        Returns:
            The generated conviction signals, bounded between [-1.0, 1.0].

        Raises:
            StrategyError: If execution fails at any point during feature
                computation or modeling.
        """
        try:
            features_config = self.manifest.get('features', [])

            # Universal Feature Calculation
            df_full, l_max = compute_all_features(raw_data, features_config)

            # Universal Warmup Purge (Protects both ML and Rule-Based from NaN lookbacks)
            df_clean = df_full.iloc[l_max:].copy()

            # Match any price normalization applied during training
            training_cfg = self.manifest.get("training", {})
            price_norm = training_cfg.get("price_normalization", "none")
            if price_norm != "none":
                df_clean = MLBridge.apply_price_normalization(
                    df_clean,
                    price_norm,
                    ffd_d=float(training_cfg.get("ffd_d", 0.4)),
                    ffd_window=int(training_cfg.get("ffd_window", 10)),
                )

            feature_ids = [f['id'] for f in features_config]
            self._audit_nans(df_clean, feature_ids)

            regime_context = self._build_regime_context(df_clean)

            # Component Initialization
            model_class, context_class = self._load_user_model_and_context()
            model = model_class()
            context = context_class() if context_class else None
            hyperparams = params if params is not None else self.manifest.get('hyperparameters', {})

            is_ml = self.manifest.get("is_ml", False)

            # Resolve artifacts: caller-provided > disk > inline training
            if artifacts is None and is_ml:
                disk_artifacts = ArtifactManager.load_artifacts(self.strategy_dir)
                if disk_artifacts:
                    artifacts = disk_artifacts
                    logger.info("Loaded persisted artifacts for inference.")

            if artifacts is not None:
                # Inference mode: apply saved scaler and skip training
                if is_ml:
                    feature_cols = [
                        c for c in df_clean.columns
                        if c.lower() not in {"open", "high", "low", "close", "volume"}
                    ]
                    df_clean = MLBridge.prepare_inference_matrix(
                        df_clean, feature_cols, l_max=0, artifacts=artifacts
                    )
                raw_signals = self._call_generate_signals(
                    model, df_clean, context, hyperparams, artifacts, regime_context
                )
            elif is_ml:
                # ML strategy without pre-trained artifacts: temporal split
                # to prevent training and predicting on the same data.
                split_point = int(len(df_clean) * 0.8)
                df_train = df_clean.iloc[:split_point].copy()
                df_eval = df_clean  # generate signals on full range

                feature_cols = [
                    c for c in df_clean.columns
                    if c.lower() not in {"open", "high", "low", "close", "volume"}
                ]

                # Fit scaler on train split only
                df_train_scaled, scaler = MLBridge.prepare_training_matrix(
                    df_train, feature_cols, l_max=0
                )

                artifacts = model.train(df_train_scaled, context, hyperparams)
                artifacts["system_scaler"] = scaler

                # Apply saved scaler to full dataset for signal generation
                df_eval = MLBridge.prepare_inference_matrix(
                    df_eval, feature_cols, l_max=0, artifacts=artifacts
                )

                logger.warning(
                    "ML backtest without saved artifacts: trained on first 80%% "
                    "of data. Run TRAIN mode for proper cross-validated results."
                )
                raw_signals = self._call_generate_signals(
                    model, df_eval, context, hyperparams, artifacts, regime_context
                )
            else:
                # Rule-based strategy: train() returns static artifacts,
                # no data leakage risk from in-sample signal generation.
                artifacts = model.train(df_clean, context, hyperparams)
                raw_signals = self._call_generate_signals(
                    model, df_clean, context, hyperparams, artifacts, regime_context
                )

            comp_mode = self.manifest.get('compression_mode', 'clip')

            clean_signals = SignalValidator.validate_and_compress(
                raw_signals=raw_signals,
                target_index=df_clean.index,
                compression_mode=comp_mode
            )

            return clean_signals

        except Exception as e:
            logger.error(f"Backtest execution failed: {e}", exc_info=True)
            raise StrategyError(f"Backtest run failed: {e}")

    def run_grid_search(self, raw_data: pd.DataFrame, param_bounds: Optional[Dict[str, List[Any]]] = None) -> List[pd.Series]:
        """
        Executes a parameter sweep across defined hyperparameter bounds.

        Args:
            raw_data (pd.DataFrame): The raw OHLCV market data.
            param_bounds (Optional[Dict[str, List[Any]]]): Dictionary of lists containing
                parameter options. Defaults to the bounds in `manifest.json`.

        Returns:
            List[pd.Series]: A list of signal series, one for each parameter permutation.
        """
        try:
            if param_bounds is None:
                param_bounds = self.manifest.get('parameter_bounds', {})

            if not param_bounds:
                return [self.run(raw_data)]

            features_config = self.manifest.get('features', [])
            df_full, l_max = compute_all_features(raw_data, features_config)

            # Apply Universal Warmup Purge
            df_clean = df_full.iloc[l_max:].copy()

            feature_ids = [f['id'] for f in features_config]
            self._audit_nans(df_clean, feature_ids)

            model_class, context_class = self._load_user_model_and_context()

            keys = list(param_bounds.keys())
            values = list(param_bounds.values())
            permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]

            logger.info(f"Starting parameter sweep across {len(permutations)} permutations.")
            results = []
            for i, p in enumerate(permutations):
                # Instantiate fresh objects to prevent state leakage
                model = model_class()
                context = context_class() if context_class else None

                artifacts = model.train(df_clean, context, p)
                signals = model.generate_signals(df_clean, context, p, artifacts)

                param_str = ", ".join([f"{k}={v}" for k, v in p.items()])
                signals.name = f"{os.path.basename(self.strategy_dir)} ({param_str})"
                results.append(signals)

            return results
        except Exception as e:
            logger.error(f"Grid search failed: {e}", exc_info=True)
            raise StrategyError(f"Grid search failed: {e}")

    def run_batch(
        self,
        datasets: Dict[str, pd.DataFrame],
        params: Optional[Dict[str, Any]] = None,
        artifacts: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, pd.Series]:
        """Executes a batch of backtests across multiple assets efficiently.

        When ``artifacts`` are provided, each asset is run in inference mode
        (no inline training, saved scaler applied for ML strategies).

        Args:
            datasets: A dictionary mapping ticker symbols to their respective
                raw OHLCV DataFrames.
            params: Strategy hyperparameters.
            artifacts: Pre-trained artifacts to use for all assets. When
                ``None``, each asset trains inline or loads from disk.

        Returns:
            A dictionary mapping ticker symbols to their generated signals.
        """
        results = {}
        if not datasets:
            logger.warning("No datasets provided for batch backtest.")
            return results

        try:
            # Load the user's classes ONCE for the entire batch
            model_class, context_class = self._load_user_model_and_context()
            features_config = self.manifest.get('features', [])
            feature_ids = [f['id'] for f in features_config]
            hyperparams = params if params is not None else self.manifest.get('hyperparameters', {})
            is_ml = self.manifest.get("is_ml", False)

            # Resolve artifacts once for the batch
            batch_artifacts = artifacts
            if batch_artifacts is None and is_ml:
                disk_artifacts = ArtifactManager.load_artifacts(self.strategy_dir)
                if disk_artifacts:
                    batch_artifacts = disk_artifacts
                    logger.info("Loaded persisted artifacts for batch inference.")

            training_cfg = self.manifest.get("training", {})
            price_norm = training_cfg.get("price_normalization", "none")
            ffd_d_cfg = float(training_cfg.get("ffd_d", 0.4))
            ffd_window_cfg = int(training_cfg.get("ffd_window", 10))

            if self.manifest.get("cross_sectional", False):
                # Phase 1: compute features for all tickers simultaneously
                processed: Dict[str, pd.DataFrame] = {}
                for ticker, df_raw in datasets.items():
                    try:
                        df_full, l_max = compute_all_features(df_raw, features_config)
                        df_clean = df_full.iloc[l_max:].copy()
                        if price_norm != "none":
                            df_clean = MLBridge.apply_price_normalization(
                                df_clean, price_norm,
                                ffd_d=ffd_d_cfg, ffd_window=ffd_window_cfg,
                            )
                        self._audit_nans(df_clean, feature_ids)
                        processed[ticker] = df_clean
                    except Exception as e:
                        logger.error(f"Feature computation failed for {ticker}: {e}", exc_info=True)

                if not processed:
                    return results

                # Phase 2: single model call with the full universe
                model = model_class()
                context = context_class() if context_class else None
                if batch_artifacts is not None:
                    raw_result = self._call_generate_signals_cross_sectional(
                        model, processed, context, hyperparams, batch_artifacts
                    )
                else:
                    inline_artifacts = model.train(processed, context, hyperparams)
                    raw_result = self._call_generate_signals_cross_sectional(
                        model, processed, context, hyperparams, inline_artifacts
                    )

                # Phase 3: validate each ticker's signals independently
                comp_mode = self.manifest.get('compression_mode', 'clip')
                for ticker, raw_signals in raw_result.items():
                    if ticker in processed:
                        try:
                            results[ticker] = SignalValidator.validate_and_compress(
                                raw_signals, processed[ticker].index, comp_mode
                            )
                        except Exception as e:
                            logger.error(f"Signal validation failed for {ticker}: {e}", exc_info=True)
                            results[ticker] = pd.Series(dtype=float)
                return results

            for ticker, df_raw in datasets.items():
                logger.info(f"Processing batch execution for {ticker}")
                try:
                    df_full, l_max = compute_all_features(df_raw, features_config)

                    # Apply Universal Warmup Purge
                    df_clean = df_full.iloc[l_max:].copy()

                    # Match any price normalization applied during training
                    if price_norm != "none":
                        df_clean = MLBridge.apply_price_normalization(
                            df_clean, price_norm,
                            ffd_d=ffd_d_cfg, ffd_window=ffd_window_cfg,
                        )

                    self._audit_nans(df_clean, feature_ids)

                    regime_context = self._build_regime_context(df_clean)

                    # Instantiate fresh objects to prevent state leakage between assets
                    model = model_class()
                    context = context_class() if context_class else None

                    if batch_artifacts is not None:
                        # Inference mode
                        if is_ml:
                            feature_cols = [
                                c for c in df_clean.columns
                                if c.lower() not in {"open", "high", "low", "close", "volume"}
                            ]
                            df_clean = MLBridge.prepare_inference_matrix(
                                df_clean, feature_cols, l_max=0,
                                artifacts=batch_artifacts,
                            )
                        signals = self._call_generate_signals(
                            model, df_clean, context, hyperparams, batch_artifacts,
                            regime_context,
                        )
                    elif is_ml:
                        # ML without artifacts: temporal split to avoid
                        # training and predicting on the same data.
                        feature_cols = [
                            c for c in df_clean.columns
                            if c.lower() not in {"open", "high", "low", "close", "volume"}
                        ]
                        split_point = int(len(df_clean) * 0.8)
                        df_train = df_clean.iloc[:split_point].copy()

                        df_train_scaled, scaler = MLBridge.prepare_training_matrix(
                            df_train, feature_cols, l_max=0
                        )
                        inline_artifacts = model.train(
                            df_train_scaled, context, hyperparams
                        )
                        inline_artifacts["system_scaler"] = scaler

                        df_eval = MLBridge.prepare_inference_matrix(
                            df_clean, feature_cols, l_max=0,
                            artifacts=inline_artifacts,
                        )
                        signals = self._call_generate_signals(
                            model, df_eval, context, hyperparams, inline_artifacts,
                            regime_context,
                        )
                    else:
                        # Rule-based: no data leakage risk
                        inline_artifacts = model.train(df_clean, context, hyperparams)
                        signals = self._call_generate_signals(
                            model, df_clean, context, hyperparams, inline_artifacts,
                            regime_context,
                        )

                    results[ticker] = signals
                except Exception as e:
                    logger.error(f"Batch execution failed for {ticker}: {e}", exc_info=True)
                    results[ticker] = pd.Series(dtype=float)

            return results

        except Exception as e:
            logger.error(f"Batch setup failed: {e}", exc_info=True)
            raise StrategyError(f"Batch run failed: {e}")


# ---------------------------------------------------------------------------
# SignalValidator
# ---------------------------------------------------------------------------

class SignalValidator:
    """
    The final safety boundary before signals reach the execution or backtesting engine.

    This class coerces arbitrary user outputs into a strict, mathematically bounded
    Phase 3 Signal Array. It guarantees that the output is a pandas Series, exactly
    matches the input timeframe's index, contains no invalid data (NaN/Inf), and
    is strictly bounded between [-1.0, 1.0].
    """

    @staticmethod
    def validate_and_compress(
        raw_signals: Any,
        target_index: pd.Index,
        compression_mode: str = 'clip'
    ) -> pd.Series:
        """
        Formats, aligns, and mathematically compresses arbitrary model predictions.

        Args:
            raw_signals (Any): The raw output from the user's `generate_signals()` method.
                Could be a list, numpy array, or pandas Series.
            target_index (pd.Index): The exact datetime index of the DataFrame that
                was passed into the user's model. Used to ensure alignment.
            compression_mode (str, optional): The mathematical method used to squash
                the signals into the [-1.0, 1.0] boundary.
                - 'clip': Hard boundary. Values > 1 become 1, values < -1 become -1.
                - 'tanh': Soft squashing function. Best for unbounded regression outputs.
                - 'probability': Maps a [0.0, 1.0] classifier probability to [-1.0, 1.0].
                Defaults to 'clip'.

        Returns:
            pd.Series: A perfectly aligned series bounded between [-1.0, 1.0], with
                0.0 indicating "no conviction / flat".

        Raises:
            StrategyError: If the output cannot be coerced into a numerical Series.

        Example:
            >>> raw = model.generate_signals(df)
            >>> clean_signals = SignalValidator.validate_and_compress(raw, df.index, 'tanh')
        """
        # 1. Type Coercion
        try:
            if not isinstance(raw_signals, pd.Series):
                signals = pd.Series(raw_signals)
            else:
                signals = raw_signals.copy()
        except Exception as e:
            logger.error(f"Could not convert raw signals to pandas Series: {e}", exc_info=True)
            raise StrategyError("User model must return a list, numpy array, or pandas Series.")

        # 2. Convert types to float, coercing strings/garbage to NaN
        signals = pd.to_numeric(signals, errors='coerce')

        # 3. Index Alignment
        if len(signals) == len(target_index) and signals.index.equals(target_index):
            pass # Perfectly aligned already
        elif len(signals) == len(target_index):
            # Same length, wrong index (user returned a list or numpy array)
            signals.index = target_index
            logger.debug("Forced target index onto user signals.")
        else:
            # User dropped rows or messed up the shape.
            logger.warning(f"Signal length ({len(signals)}) does not match input length ({len(target_index)}). Attempting to reindex.")
            # If they provided a proper index, we map it. If not, this will likely fail or fill with NaNs.
            try:
                signals = signals.reindex(target_index)
            except Exception as e:
                raise StrategyError(f"Critical index mismatch. Cannot align signals to input data. {e}")

        # 4. Handle Inf and NaN before math operations
        signals = signals.replace([np.inf, -np.inf], np.nan)
        signals = signals.fillna(0.0) # NaN implies no conviction (0 weight)

        # 5. Mathematical Compression to [-1.0, 1.0]
        if compression_mode == 'clip':
            signals = signals.clip(lower=-1.0, upper=1.0)

        elif compression_mode == 'tanh':
            # Great for raw expected returns (e.g., passing 0.05 return through tanh)
            # You might need a scalar multiplier here depending on asset volatility
            signals = np.tanh(signals)

        elif compression_mode == 'probability':
            # Maps XGBoost binary probabilities [0, 1] to [-1, 1]
            # 0.5 becomes 0.0 (neutral). 1.0 becomes 1.0 (long). 0.0 becomes -1.0 (short).
            signals = (signals * 2) - 1.0

        else:
            logger.warning(f"Unknown compression mode '{compression_mode}'. Defaulting to 'clip'.")
            signals = signals.clip(lower=-1.0, upper=1.0)

        # Final safety check
        signals.name = "conviction_signal"
        return signals
