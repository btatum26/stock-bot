"""Dedicated training pipeline for strategy model development.

This module owns the data pipeline surrounding a strategy's ``train()`` method.
It does NOT implement training logic itself — that belongs to the user's
``SignalModel`` subclass. Instead, it controls feature computation, data
splitting, scaling, validation, and artifact persistence.
"""

import os
import json
import math
import sys
import importlib.util
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Union
from sklearn.preprocessing import MinMaxScaler

from .features.features import compute_all_features
from .ml_bridge.orchestrator import MLBridge
from .ml_bridge.artifact_manager import ArtifactManager
from .optimization.cpcv_splitter import CPCVSplitter
from .logger import logger
from .exceptions import StrategyError


class LocalTrainer:
    """Controls data preparation and validation for strategy training.

    This class owns the data pipeline surrounding a strategy's ``train()``
    method. The strategy itself handles all model logic; the trainer controls:

    - Feature computation and warmup purge
    - Train/validation splitting (temporal or CPCV)
    - Feature scaling for ML strategies (via ``MLBridge``)
    - Validation metric calculation (via ``Tearsheet``)
    - Artifact persistence (via ``ArtifactManager``)

    Attributes:
        strategy_dir: Normalized path to the strategy folder.
        manifest: Parsed manifest.json dictionary.
        training_config: Merged training configuration (manifest defaults
            overlaid on class defaults).
        is_ml: Whether the strategy requires ML preprocessing (scaling).
    """

    OHLCV_COLS = frozenset({"open", "high", "low", "close", "volume"})

    DEFAULT_TRAINING_CONFIG = {
        "split_method": "cpcv",
        "train_ratio": 0.8,
        "n_groups": 6,
        "k_test_groups": 2,
        "embargo_pct": 0.01,
        # Multi-ticker training uses walk-forward splits by calendar date.
        # TODO: Replace with panel-aware CPCV. Current walk-forward is
        # robust to cross-sectional leakage but less sample-efficient,
        # and a single temporal split can still overfit to regime in
        # any one validation window — multiple folds mitigate this.
        "n_walk_forward_folds": 5,
        "price_normalization": "none",
    }

    def __init__(self, strategy_dir: str):
        """Initializes the trainer for a specific strategy directory.

        Args:
            strategy_dir: Path to the strategy folder containing
                ``manifest.json``, ``model.py``, and ``context.py``.

        Raises:
            StrategyError: If the manifest file cannot be found or parsed.
        """
        self.strategy_dir = os.path.normpath(strategy_dir)
        manifest_path = os.path.join(self.strategy_dir, "manifest.json")
        try:
            with open(manifest_path, "r") as f:
                self.manifest = json.load(f)
        except Exception as e:
            raise StrategyError(
                f"Missing or invalid manifest in {self.strategy_dir}: {e}"
            )

        self.training_config = {
            **self.DEFAULT_TRAINING_CONFIG,
            **self.manifest.get("training", {}),
        }
        self.is_ml = self.manifest.get("is_ml", False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        raw_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Executes the full training pipeline.

        Accepts either a single DataFrame (legacy single-ticker path) or
        a dict of ``{ticker: DataFrame}`` for multi-ticker pooled training.
        A one-element dict is treated as the single-ticker case.

        Args:
            raw_data: Raw OHLCV DataFrame, or dict mapping ticker symbols
                to raw OHLCV DataFrames. Each frame has a ``DatetimeIndex``.
            params: Hyperparameters. Falls back to the manifest
                ``hyperparameters`` section when not provided.

        Returns:
            Dictionary containing ``train_metrics``, ``val_metrics``,
            ``split_info``, and ``params``.
        """
        if isinstance(raw_data, dict):
            if not raw_data:
                raise StrategyError("No datasets provided to trainer")
            if len(raw_data) == 1:
                only_df = next(iter(raw_data.values()))
                return self._run_single(only_df, params)
            return self._run_multi(raw_data, params)
        return self._run_single(raw_data, params)

    # ------------------------------------------------------------------
    # Single-ticker pipeline (legacy path)
    # ------------------------------------------------------------------

    def _run_single(
        self,
        raw_data: pd.DataFrame,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Single-ticker training flow.

        Pipeline stages:
            1. Compute features and purge warmup NaNs.
            2. Split into train/validation folds (temporal or CPCV).
            3. For each fold: optionally scale, train, validate.
            4. For multi-fold (CPCV): retrain on full dataset.
            5. Save final artifacts to disk.
        """
        # 1. Compute features
        features_config = self.manifest.get("features", [])
        df_full, l_max = compute_all_features(raw_data, features_config)
        df_clean = df_full.iloc[l_max:].copy()

        # Apply price normalization before the model sees OHLCV columns.
        # Must happen after feature computation (features use raw prices) and
        # after warmup purge (so log-diff NaN drop only costs 1 row).
        price_norm = self.training_config.get("price_normalization", "none")
        if price_norm != "none":
            df_clean = MLBridge.apply_price_normalization(df_clean, price_norm)
            logger.info(f"Price normalization applied: {price_norm}")

        # Identify computed feature columns (everything beyond raw OHLCV)
        feature_cols = [
            c for c in df_clean.columns if c.lower() not in self.OHLCV_COLS
        ]
        self._audit_features(df_clean, features_config)

        # 2. Load model and context
        model_class, context_class = self._load_user_model_and_context()
        hyperparams = (
            params
            if params is not None
            else self.manifest.get("hyperparameters", {})
        )

        # 3. Generate splits
        folds = self._generate_splits(df_clean, l_max)

        # 4. Train and evaluate each fold
        fold_results: List[Dict[str, Any]] = []
        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            logger.info(
                f"Training fold {fold_idx + 1}/{len(folds)} "
                f"(train={len(train_idx)}, val={len(val_idx)})"
            )
            result = self._train_fold(
                model_class=model_class,
                context_class=context_class,
                df_clean=df_clean,
                raw_data=raw_data,
                train_idx=train_idx,
                val_idx=val_idx,
                feature_cols=feature_cols,
                hyperparams=hyperparams,
            )
            fold_results.append(result)

        # 5. Final artifact selection
        if len(folds) > 1:
            # CPCV: folds were for evaluation only — retrain on full data
            logger.info(
                "CPCV validation complete. Retraining on full dataset "
                "for final artifacts."
            )
            final = self._train_final(
                model_class, context_class, df_clean, feature_cols, hyperparams
            )
            artifacts = final["artifacts"]
        else:
            artifacts = fold_results[0]["artifacts"]

        # 6. Persist artifacts
        ArtifactManager.save_artifacts(self.strategy_dir, artifacts)

        # 7. Build and return report
        results = self._build_results(fold_results, folds, df_clean, hyperparams)
        feature_analysis = artifacts.get("feature_analysis")
        if feature_analysis:
            self._print_feature_analysis(feature_analysis)
            results["feature_analysis"] = feature_analysis
        return results

    # ------------------------------------------------------------------
    # Data splitting
    # ------------------------------------------------------------------

    def _generate_splits(
        self, df: pd.DataFrame, l_max: int
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Creates train/validation index splits based on the configured method.

        Args:
            df: The warmup-purged feature DataFrame.
            l_max: Maximum feature lookback window, used for purge gap
                sizing in CPCV.

        Returns:
            List of ``(train_indices, val_indices)`` tuples. Each element
            is a numpy array of integer positional indices into ``df``.
        """
        method = self.training_config["split_method"]

        if method == "temporal":
            return self._split_temporal(df)
        elif method == "cpcv":
            splitter = CPCVSplitter(
                n_groups=self.training_config["n_groups"],
                k_test_groups=self.training_config["k_test_groups"],
                embargo_pct=self.training_config["embargo_pct"],
            )
            return splitter.split(df, l_max)
        else:
            logger.warning(
                f"Unknown split method '{method}'. Falling back to temporal."
            )
            return self._split_temporal(df)

    def _split_temporal(
        self, df: pd.DataFrame
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Performs a simple chronological train/validation split.

        Training data comes strictly before validation data. No shuffling
        is applied, preserving the temporal ordering required for
        time-series models.

        Args:
            df: The feature DataFrame to split.

        Returns:
            Single-element list containing one ``(train_idx, val_idx)``
            tuple of positional index arrays.
        """
        n = len(df)
        ratio = self.training_config["train_ratio"]
        split_point = int(n * ratio)

        train_idx = np.arange(0, split_point)
        val_idx = np.arange(split_point, n)

        return [(train_idx, val_idx)]

    # ------------------------------------------------------------------
    # Fold training
    # ------------------------------------------------------------------

    def _train_fold(
        self,
        model_class: type,
        context_class: Optional[type],
        df_clean: pd.DataFrame,
        raw_data: pd.DataFrame,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        feature_cols: List[str],
        hyperparams: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Trains on one fold and evaluates on held-out validation data.

        Handles ML scaling when ``is_ml`` is set in the manifest. The
        scaler is fitted exclusively on the training split and applied
        (transform-only) to the validation split to prevent information
        leakage.

        Args:
            model_class: The user's ``SignalModel`` subclass (uninstantiated).
            context_class: The user's ``Context`` class, or ``None``.
            df_clean: Full warmup-purged feature DataFrame.
            raw_data: Original OHLCV DataFrame (for metric calculation).
            train_idx: Positional indices for the training set.
            val_idx: Positional indices for the validation set.
            feature_cols: List of computed feature column names to scale.
            hyperparams: Strategy hyperparameters.

        Returns:
            Dictionary with keys ``artifacts``, ``train_metrics``, and
            ``val_metrics``.
        """
        from .backtester import Tearsheet, SignalValidator

        df_train = df_clean.iloc[train_idx].copy()
        df_val = df_clean.iloc[val_idx].copy()

        # Scale features for ML strategies
        scaler = None
        if self.is_ml and feature_cols:
            # l_max=0 because data is already warmup-purged
            df_train, scaler = MLBridge.prepare_training_matrix(
                df_train, feature_cols, l_max=0
            )
            df_val = MLBridge.prepare_inference_matrix(
                df_val, feature_cols, l_max=0,
                artifacts={"system_scaler": scaler},
            )

        # Train
        model = model_class()
        context = context_class() if context_class else None
        if self.is_ml:
            artifacts = self._fit_ml_single(
                model, context, df_train, feature_cols, hyperparams
            )
        else:
            artifacts = model.train(df_train, context, hyperparams)

        # Embed scaler in artifacts for downstream inference
        if scaler is not None:
            artifacts["system_scaler"] = scaler

        # Evaluate
        comp_mode = self.manifest.get("compression_mode", "clip")

        val_signals = model.generate_signals(
            df_val, context, hyperparams, artifacts
        )
        val_signals = SignalValidator.validate_and_compress(
            val_signals, df_val.index, comp_mode
        )
        raw_val = raw_data.loc[raw_data.index.isin(df_val.index)]
        val_metrics = Tearsheet.calculate_metrics(raw_val, val_signals)

        train_signals = model.generate_signals(
            df_train, context, hyperparams, artifacts
        )
        train_signals = SignalValidator.validate_and_compress(
            train_signals, df_train.index, comp_mode
        )
        raw_train = raw_data.loc[raw_data.index.isin(df_train.index)]
        train_metrics = Tearsheet.calculate_metrics(raw_train, train_signals)

        return {
            "artifacts": artifacts,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }

    def _train_final(
        self,
        model_class: type,
        context_class: Optional[type],
        df_clean: pd.DataFrame,
        feature_cols: List[str],
        hyperparams: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Retrains on the full dataset after CPCV validation.

        After cross-validation confirms the strategy generalizes, this
        method trains a final model on all available data to maximize the
        information used for live deployment.

        Args:
            model_class: The user's ``SignalModel`` subclass.
            context_class: The user's ``Context`` class, or ``None``.
            df_clean: Full warmup-purged feature DataFrame.
            feature_cols: Computed feature column names.
            hyperparams: Strategy hyperparameters.

        Returns:
            Dictionary with an ``artifacts`` key containing the final
            trained artifacts (including the scaler, if ML).
        """
        df_full = df_clean.copy()
        scaler = None

        if self.is_ml and feature_cols:
            df_full, scaler = MLBridge.prepare_training_matrix(
                df_full, feature_cols, l_max=0
            )

        model = model_class()
        context = context_class() if context_class else None
        if self.is_ml:
            artifacts = self._fit_ml_single(
                model, context, df_full, feature_cols, hyperparams
            )
        else:
            artifacts = model.train(df_full, context, hyperparams)

        if scaler is not None:
            artifacts["system_scaler"] = scaler

        return {"artifacts": artifacts}

    def _fit_ml_single(
        self,
        model,
        context,
        df: pd.DataFrame,
        feature_cols: List[str],
        hyperparams: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Fits an ML model via build_labels + fit_model on a single df.

        Used by the single-ticker path. The multi-ticker path builds
        labels per-ticker then pools before calling ``fit_model`` once.
        """
        y = model.build_labels(df, context, hyperparams)
        if not isinstance(y, pd.Series):
            raise StrategyError(
                "build_labels must return a pandas Series aligned to df.index"
            )
        mask = y.notna()
        df_valid = df.loc[mask]
        y_valid = y.loc[mask]

        X = df_valid[feature_cols]
        y_arr = y_valid.to_numpy()

        artifacts = model.fit_model(X, y_arr, hyperparams)
        if not isinstance(artifacts, dict):
            raise StrategyError("fit_model must return a dict of artifacts")
        artifacts["feature_cols"] = feature_cols
        artifacts["feature_analysis"] = self._compute_feature_analysis(
            X, y_valid, artifacts.get("model"), feature_cols
        )
        return artifacts

    @staticmethod
    def _compute_feature_analysis(
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        model: Any,
        feature_cols: List[str],
    ) -> List[Dict[str, Any]]:
        """Computes per-feature diagnostics on the pooled training matrix.

        Returns a list of dicts (one per feature) with:
            - ``name``: feature column name
            - ``importance``: model.feature_importances_ if available, else None
            - ``pearson``: Pearson correlation of feature with y
            - ``spearman``: Spearman rank correlation of feature with y
            - ``mi``: mutual information with y (classification)

        Captures both linear (pearson), monotonic (spearman), and
        nonlinear (MI) dependence so the user can see which features
        actually carry signal the model can exploit.
        """
        y_series = pd.Series(y, index=X.index) if not isinstance(y, pd.Series) else y
        y_arr = y_series.to_numpy()

        importances: Optional[np.ndarray] = None
        if model is not None and hasattr(model, "feature_importances_"):
            fi = getattr(model, "feature_importances_")
            if fi is not None and len(fi) == len(feature_cols):
                importances = np.asarray(fi, dtype=float)

        mi_scores: Optional[np.ndarray] = None
        try:
            from sklearn.feature_selection import mutual_info_classif
            y_int = y_arr.astype(int)
            mi_scores = mutual_info_classif(
                X.to_numpy(), y_int, random_state=42, discrete_features=False
            )
        except Exception as e:
            logger.warning(f"Mutual information failed: {e}")

        rows: List[Dict[str, Any]] = []
        for i, col in enumerate(feature_cols):
            x_col = X[col]
            try:
                pearson = float(x_col.corr(y_series, method="pearson"))
            except Exception:
                pearson = float("nan")
            try:
                spearman = float(x_col.corr(y_series, method="spearman"))
            except Exception:
                spearman = float("nan")

            rows.append({
                "name": col,
                "importance": (
                    float(importances[i]) if importances is not None else None
                ),
                "pearson": pearson,
                "spearman": spearman,
                "mi": (
                    float(mi_scores[i]) if mi_scores is not None else None
                ),
            })

        # Sort by importance if present, else by |pearson|.
        if importances is not None:
            rows.sort(key=lambda r: -(r["importance"] or 0.0))
        else:
            rows.sort(key=lambda r: -abs(r["pearson"]) if not np.isnan(r["pearson"]) else 0.0)

        return rows

    @staticmethod
    def _print_feature_analysis(analysis: List[Dict[str, Any]]) -> None:
        """Prints the feature importance / correlation table."""
        if not analysis:
            return

        has_importance = any(r["importance"] is not None for r in analysis)
        has_mi = any(r["mi"] is not None for r in analysis)

        print("\n" + "=" * 78)
        print(" " * 25 + "FEATURE ANALYSIS (pooled train)")
        print("=" * 78)
        header = f"  {'Feature':<36}"
        if has_importance:
            header += f" {'Importance':>11}"
        header += f" {'Pearson':>9} {'Spearman':>10}"
        if has_mi:
            header += f" {'MI':>8}"
        print(header)
        print("  " + "-" * 76)

        for r in analysis:
            line = f"  {r['name'][:36]:<36}"
            if has_importance:
                imp = r["importance"]
                line += f" {imp:>11.4f}" if imp is not None else f" {'N/A':>11}"
            p = r["pearson"]
            s = r["spearman"]
            line += (
                f" {p:>9.3f}" if not (isinstance(p, float) and np.isnan(p))
                else f" {'N/A':>9}"
            )
            line += (
                f" {s:>10.3f}" if not (isinstance(s, float) and np.isnan(s))
                else f" {'N/A':>10}"
            )
            if has_mi:
                mi = r["mi"]
                line += f" {mi:>8.4f}" if mi is not None else f" {'N/A':>8}"
            print(line)

        print("=" * 78 + "\n")

    # ------------------------------------------------------------------
    # Results aggregation
    # ------------------------------------------------------------------

    def _build_results(
        self,
        fold_results: List[Dict[str, Any]],
        folds: List[Tuple[np.ndarray, np.ndarray]],
        df_clean: pd.DataFrame,
        hyperparams: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Aggregates fold results into the final training report.

        For single-fold (temporal) splits, returns metrics directly. For
        multi-fold (CPCV) splits, computes mean and standard deviation
        across folds for each scalar metric.

        Args:
            fold_results: List of per-fold result dictionaries.
            folds: The train/val index splits that were used.
            df_clean: The warmup-purged feature DataFrame (for date ranges).
            hyperparams: The hyperparameters that were used.

        Returns:
            Aggregated results dictionary with ``train_metrics``,
            ``val_metrics``, ``split_info``, and ``params``.
        """
        skip_keys = {"equity_curve", "portfolio", "bh_portfolio", "trade_log"}

        if len(fold_results) == 1:
            r = fold_results[0]
            train_scalar = {
                k: v for k, v in r["train_metrics"].items() if k not in skip_keys
            }
            val_scalar = {
                k: v for k, v in r["val_metrics"].items() if k not in skip_keys
            }
            train_idx, val_idx = folds[0]
            split_info = {
                "method": "temporal",
                "train_size": len(train_idx),
                "val_size": len(val_idx),
                "train_range": [
                    str(df_clean.index[train_idx[0]]),
                    str(df_clean.index[train_idx[-1]]),
                ],
                "val_range": [
                    str(df_clean.index[val_idx[0]]),
                    str(df_clean.index[val_idx[-1]]),
                ],
            }
        else:
            train_scalar = self._aggregate_fold_metrics(
                [r["train_metrics"] for r in fold_results], skip_keys
            )
            val_scalar = self._aggregate_fold_metrics(
                [r["val_metrics"] for r in fold_results], skip_keys
            )
            split_info = {
                "method": "cpcv",
                "n_folds": len(folds),
                "n_groups": self.training_config["n_groups"],
                "k_test_groups": self.training_config["k_test_groups"],
            }

        self._print_training_summary(train_scalar, val_scalar, split_info)

        return {
            "train_metrics": train_scalar,
            "val_metrics": val_scalar,
            "split_info": split_info,
            "params": hyperparams,
        }

    @staticmethod
    def _aggregate_fold_metrics(
        metrics_list: List[Dict[str, Any]], skip_keys: set
    ) -> Dict[str, Any]:
        """Computes mean and std for each scalar metric across folds.

        Args:
            metrics_list: List of metric dictionaries, one per fold.
            skip_keys: Set of keys to exclude (time-series objects).

        Returns:
            Dictionary mapping metric names to
            ``{'mean': float, 'std': float}``.
        """
        aggregated: Dict[str, Any] = {}
        scalar_keys = [k for k in metrics_list[0] if k not in skip_keys]

        for key in scalar_keys:
            values = [
                m[key]
                for m in metrics_list
                if isinstance(m.get(key), (int, float))
                and not isinstance(m.get(key), bool)
                and math.isfinite(m[key])
            ]
            if values:
                aggregated[key] = {
                    "mean": round(float(np.mean(values)), 4),
                    "std": round(float(np.std(values)), 4),
                }
        return aggregated

    @staticmethod
    def _print_training_summary(
        train_metrics: Dict[str, Any],
        val_metrics: Dict[str, Any],
        split_info: Dict[str, Any],
    ) -> None:
        """Prints a formatted training report to the console.

        Args:
            train_metrics: Scalar metrics from the training set.
            val_metrics: Scalar metrics from the validation set.
            split_info: Metadata about the data split used.
        """
        print("\n" + "=" * 55)
        print(" " * 15 + "TRAINING REPORT")
        print("=" * 55)

        # Split info
        method = split_info["method"]
        print(f"\n  Split Method: {method.upper()}")
        if method == "temporal":
            print(f"  Train Size:   {split_info['train_size']} bars")
            print(f"  Val Size:     {split_info['val_size']} bars")
            print(
                f"  Train Range:  {split_info['train_range'][0]}"
                f" -> {split_info['train_range'][1]}"
            )
            print(
                f"  Val Range:    {split_info['val_range'][0]}"
                f" -> {split_info['val_range'][1]}"
            )
        elif method == "walk_forward":
            print(f"  Folds:        {split_info['n_folds']}")
            print(f"  Tickers:      {split_info['n_tickers']}")
            print(f"  Total Rows:   {split_info['n_rows']}")
        else:
            print(f"  Folds:        {split_info['n_folds']}")
            print(
                f"  Groups:       {split_info['n_groups']} "
                f"(k={split_info['k_test_groups']})"
            )

        # Metrics table
        is_cpcv = isinstance(next(iter(val_metrics.values()), None), dict)

        print(f"\n  {'Metric':<28} {'Train':>12} {'Val':>12}")
        print("  " + "-" * 54)

        for key in val_metrics:
            if is_cpcv:
                t_val = train_metrics.get(key, {})
                v_val = val_metrics[key]
                t_str = f"{t_val.get('mean', 0):.2f}+/-{t_val.get('std', 0):.2f}"
                v_str = f"{v_val.get('mean', 0):.2f}+/-{v_val.get('std', 0):.2f}"
            else:
                t_str = f"{train_metrics.get(key, 'N/A')}"
                v_str = f"{val_metrics.get(key, 'N/A')}"
            print(f"  {key:<28} {t_str:>12} {v_str:>12}")

        print("\n" + "=" * 55)

    # ------------------------------------------------------------------
    # Multi-ticker pipeline
    # ------------------------------------------------------------------

    def _run_multi(
        self,
        datasets: Dict[str, pd.DataFrame],
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Multi-ticker pooled training flow.

        Per-ticker: computes features, purges warmup rows, applies price
        normalization. Across tickers: builds labels per ticker (so the
        user's ``build_labels`` only ever sees a single asset), then
        pools X and y across all tickers and fits one scaler and one
        model on the stacked matrix. Evaluates per-ticker and averages
        metrics within each fold, then across folds.
        """
        if not self.is_ml:
            raise StrategyError(
                "Multi-ticker training is only supported for ML strategies "
                "(is_ml: true). Rule-based strategies should pass one "
                "ticker at a time."
            )

        features_config = self.manifest.get("features", [])
        price_norm = self.training_config.get("price_normalization", "none")

        # 1. Per-ticker prep
        prepared: Dict[str, Dict[str, Any]] = {}
        feature_cols: Optional[List[str]] = None
        for ticker, raw in datasets.items():
            df_full, l_max = compute_all_features(raw, features_config)
            df_clean = df_full.iloc[l_max:].copy()
            if price_norm != "none":
                df_clean = MLBridge.apply_price_normalization(df_clean, price_norm)
            self._audit_features(df_clean, features_config)

            cols = [c for c in df_clean.columns if c.lower() not in self.OHLCV_COLS]
            if feature_cols is None:
                feature_cols = cols
            elif cols != feature_cols:
                raise StrategyError(
                    f"Feature column mismatch for {ticker}: expected "
                    f"{feature_cols}, got {cols}"
                )
            prepared[ticker] = {"df_clean": df_clean, "raw": raw}

        if feature_cols is None:
            raise StrategyError("No tickers produced any feature data")

        logger.info(
            f"Multi-ticker prep complete: {len(prepared)} tickers, "
            f"{len(feature_cols)} features."
        )

        # 2. Load model and context
        model_class, context_class = self._load_user_model_and_context()
        hyperparams = (
            params
            if params is not None
            else self.manifest.get("hyperparameters", {})
        )

        # 3. Generate walk-forward date folds
        n_folds = int(self.training_config.get("n_walk_forward_folds", 5))
        folds = self._split_walk_forward_by_date(prepared, n_folds)

        # 4. Train and evaluate each fold
        fold_results: List[Dict[str, Any]] = []
        for fold_idx, (train_start, train_end, val_start, val_end) in enumerate(folds):
            logger.info(
                f"Fold {fold_idx + 1}/{len(folds)}: "
                f"train [{train_start.date()} -> {train_end.date()}], "
                f"val [{val_start.date()} -> {val_end.date()}]"
            )
            result = self._train_fold_multi(
                model_class=model_class,
                context_class=context_class,
                prepared=prepared,
                feature_cols=feature_cols,
                hyperparams=hyperparams,
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
            )
            fold_results.append(result)

        # 5. Retrain on full pooled dataset for final artifacts
        logger.info("Walk-forward validation complete. Retraining on full pooled dataset.")
        final = self._train_final_multi(
            model_class, context_class, prepared, feature_cols, hyperparams
        )
        artifacts = final["artifacts"]

        # 6. Persist
        ArtifactManager.save_artifacts(self.strategy_dir, artifacts)

        # 7. Build report
        results = self._build_multi_results(
            fold_results, folds, prepared, hyperparams
        )
        feature_analysis = artifacts.get("feature_analysis")
        if feature_analysis:
            self._print_feature_analysis(feature_analysis)
            results["feature_analysis"] = feature_analysis
        return results

    def _split_walk_forward_by_date(
        self,
        prepared: Dict[str, Dict[str, Any]],
        n_folds: int,
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """Expanding walk-forward splits across the union of all dates.

        Each fold trains on ``[first_date, cutoff_k]`` and validates on
        ``[cutoff_k + 1 day, cutoff_{k+1}]``. Because splits are keyed
        to calendar dates (not row positions), same-date bars from
        different tickers cannot straddle the train/val boundary and
        cross-sectional leakage is avoided.

        The initial 40% of dates is reserved for the first training
        window; the remaining 60% is divided into ``n_folds`` equal
        validation windows.

        TODO: Panel-aware CPCV. Walk-forward is robust but less
        sample-efficient than CPCV with proper date-group purging.
        """
        all_dates = pd.DatetimeIndex([])
        for p in prepared.values():
            idx = pd.DatetimeIndex(p["df_clean"].index)
            all_dates = all_dates.union(idx)
        all_dates = all_dates.sort_values()

        n = len(all_dates)
        if n < n_folds * 2:
            raise StrategyError(
                f"Not enough dates ({n}) for {n_folds} walk-forward folds"
            )

        initial_train_end_idx = max(int(n * 0.4), 1)
        val_pool_size = n - initial_train_end_idx
        val_fold_size = max(val_pool_size // n_folds, 1)

        folds: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
        train_start = all_dates[0]
        for k in range(n_folds):
            train_end_idx = initial_train_end_idx + k * val_fold_size - 1
            if train_end_idx < 0:
                train_end_idx = 0
            val_start_idx = train_end_idx + 1
            if val_start_idx >= n:
                break
            val_end_idx = min(val_start_idx + val_fold_size - 1, n - 1)
            folds.append((
                train_start,
                all_dates[train_end_idx],
                all_dates[val_start_idx],
                all_dates[val_end_idx],
            ))

        if not folds:
            raise StrategyError("Walk-forward splitter produced zero folds")
        return folds

    def _train_fold_multi(
        self,
        model_class: type,
        context_class: Optional[type],
        prepared: Dict[str, Dict[str, Any]],
        feature_cols: List[str],
        hyperparams: Dict[str, Any],
        train_start: pd.Timestamp,
        train_end: pd.Timestamp,
        val_start: pd.Timestamp,
        val_end: pd.Timestamp,
    ) -> Dict[str, Any]:
        """Trains on one walk-forward fold.

        Per-ticker label construction, pooled fit, per-ticker evaluation.
        """
        from .backtester import Tearsheet, SignalValidator

        model = model_class()
        context = context_class() if context_class else None

        train_X_parts: List[pd.DataFrame] = []
        train_y_parts: List[np.ndarray] = []
        train_slices: Dict[str, pd.DataFrame] = {}
        val_slices: Dict[str, pd.DataFrame] = {}

        for ticker, p in prepared.items():
            df = p["df_clean"]
            train_mask = (df.index >= train_start) & (df.index <= train_end)
            val_mask = (df.index >= val_start) & (df.index <= val_end)
            df_train_t = df.loc[train_mask]
            df_val_t = df.loc[val_mask]

            if df_train_t.empty:
                continue

            y_t = model.build_labels(df_train_t, context, hyperparams)
            if not isinstance(y_t, pd.Series):
                raise StrategyError(
                    "build_labels must return a pandas Series"
                )
            valid = y_t.notna()
            df_train_valid = df_train_t.loc[valid]
            y_valid = y_t.loc[valid]

            if df_train_valid.empty:
                continue

            train_X_parts.append(df_train_valid[feature_cols])
            train_y_parts.append(y_valid.to_numpy())
            train_slices[ticker] = df_train_t
            val_slices[ticker] = df_val_t

        if not train_X_parts:
            raise StrategyError(
                f"Fold produced no training data [{train_start} -> {train_end}]"
            )

        X_train_df = pd.concat(train_X_parts, axis=0)
        y_train = np.concatenate(train_y_parts, axis=0)

        # Fit scaler on pooled X, keep DataFrame shape so feature names flow through to fit_model.
        scaler = MinMaxScaler(feature_range=(-1.0, 1.0))  # type: ignore[arg-type]
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train_df),
            index=X_train_df.index,
            columns=X_train_df.columns,
        )

        artifacts = model.fit_model(X_train_scaled, y_train, hyperparams)
        if not isinstance(artifacts, dict):
            raise StrategyError("fit_model must return a dict of artifacts")
        artifacts["feature_cols"] = feature_cols
        artifacts["system_scaler"] = scaler

        comp_mode = self.manifest.get("compression_mode", "clip")
        val_metrics_list: List[Dict[str, Any]] = []
        train_metrics_list: List[Dict[str, Any]] = []

        for ticker, df_val_t in val_slices.items():
            if df_val_t.empty:
                continue
            raw = prepared[ticker]["raw"]
            raw_val = raw.loc[raw.index.isin(df_val_t.index)]
            if raw_val.empty:
                continue

            df_val_scaled = df_val_t.copy()
            df_val_scaled[feature_cols] = scaler.transform(
                df_val_t[feature_cols]
            )
            signals = model.generate_signals(
                df_val_scaled, context, hyperparams, artifacts
            )
            signals = SignalValidator.validate_and_compress(
                signals, df_val_scaled.index, comp_mode
            )
            val_metrics_list.append(
                Tearsheet.calculate_metrics(raw_val, signals)
            )

        for ticker, df_train_t in train_slices.items():
            if df_train_t.empty:
                continue
            raw = prepared[ticker]["raw"]
            raw_train = raw.loc[raw.index.isin(df_train_t.index)]
            if raw_train.empty:
                continue

            df_train_scaled = df_train_t.copy()
            df_train_scaled[feature_cols] = scaler.transform(
                df_train_t[feature_cols]
            )
            signals = model.generate_signals(
                df_train_scaled, context, hyperparams, artifacts
            )
            signals = SignalValidator.validate_and_compress(
                signals, df_train_scaled.index, comp_mode
            )
            train_metrics_list.append(
                Tearsheet.calculate_metrics(raw_train, signals)
            )

        return {
            "artifacts": artifacts,
            "train_metrics": self._average_ticker_metrics(train_metrics_list),
            "val_metrics": self._average_ticker_metrics(val_metrics_list),
        }

    def _train_final_multi(
        self,
        model_class: type,
        context_class: Optional[type],
        prepared: Dict[str, Dict[str, Any]],
        feature_cols: List[str],
        hyperparams: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Retrains on the full pooled dataset (all tickers, all dates).

        Called after walk-forward validation to produce the artifacts
        that get persisted for live inference.
        """
        model = model_class()
        context = context_class() if context_class else None

        X_parts: List[pd.DataFrame] = []
        y_parts: List[np.ndarray] = []
        for p in prepared.values():
            df = p["df_clean"]
            y = model.build_labels(df, context, hyperparams)
            if not isinstance(y, pd.Series):
                raise StrategyError(
                    "build_labels must return a pandas Series"
                )
            valid = y.notna()
            df_valid = df.loc[valid]
            if df_valid.empty:
                continue
            X_parts.append(df_valid[feature_cols])
            y_parts.append(y.loc[valid].to_numpy())

        if not X_parts:
            raise StrategyError("Final training produced no data across all tickers")

        X_df = pd.concat(X_parts, axis=0)
        y_full = np.concatenate(y_parts, axis=0)

        scaler = MinMaxScaler(feature_range=(-1.0, 1.0))  # type: ignore[arg-type]
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X_df),
            index=X_df.index,
            columns=X_df.columns,
        )

        artifacts = model.fit_model(X_scaled, y_full, hyperparams)
        if not isinstance(artifacts, dict):
            raise StrategyError("fit_model must return a dict of artifacts")
        artifacts["feature_cols"] = feature_cols
        artifacts["system_scaler"] = scaler
        artifacts["feature_analysis"] = self._compute_feature_analysis(
            X_scaled, y_full, artifacts.get("model"), feature_cols
        )

        return {"artifacts": artifacts}

    @staticmethod
    def _average_ticker_metrics(
        metrics_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Averages scalar tearsheet metrics across tickers (within a fold)."""
        skip_keys = {"equity_curve", "portfolio", "bh_portfolio", "trade_log"}
        if not metrics_list:
            return {}
        keys = [k for k in metrics_list[0] if k not in skip_keys]
        out: Dict[str, Any] = {}
        for k in keys:
            values = [
                m[k]
                for m in metrics_list
                if isinstance(m.get(k), (int, float))
                and not isinstance(m.get(k), bool)
                and math.isfinite(m[k])
            ]
            if values:
                out[k] = round(float(np.mean(values)), 4)
        return out

    def _build_multi_results(
        self,
        fold_results: List[Dict[str, Any]],
        folds: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]],
        prepared: Dict[str, Dict[str, Any]],
        hyperparams: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Aggregates walk-forward fold results into the final report."""
        skip_keys = {"equity_curve", "portfolio", "bh_portfolio", "trade_log"}

        train_scalar = self._aggregate_fold_metrics(
            [r["train_metrics"] for r in fold_results], skip_keys
        )
        val_scalar = self._aggregate_fold_metrics(
            [r["val_metrics"] for r in fold_results], skip_keys
        )

        n_rows = sum(len(p["df_clean"]) for p in prepared.values())
        split_info = {
            "method": "walk_forward",
            "n_folds": len(folds),
            "n_tickers": len(prepared),
            "n_rows": n_rows,
            "tickers": list(prepared.keys()),
            "fold_ranges": [
                {
                    "train": [str(f[0]), str(f[1])],
                    "val": [str(f[2]), str(f[3])],
                }
                for f in folds
            ],
        }

        self._print_training_summary(train_scalar, val_scalar, split_info)

        return {
            "train_metrics": train_scalar,
            "val_metrics": val_scalar,
            "split_info": split_info,
            "params": hyperparams,
        }

    # ------------------------------------------------------------------
    # Feature auditing
    # ------------------------------------------------------------------

    def _audit_features(
        self, df: pd.DataFrame, features_config: List[Dict[str, Any]]
    ) -> None:
        """Scans for NaN values in computed features and logs warnings.

        Since the ``l_max`` purge handles rolling indicator warmup periods,
        any NaNs caught here indicate flawed data or broken indicators.

        Args:
            df: The warmup-purged DataFrame.
            features_config: Feature configuration list from the manifest.
        """
        for feat in features_config:
            fid = feat.get("id", "")
            cols = [c for c in df.columns if c.startswith(fid)]
            for col in cols:
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    logger.warning(
                        f"Feature '{col}' has {nan_count} unexpected NaNs "
                        f"after warmup purge."
                    )

    # ------------------------------------------------------------------
    # Dynamic model loading
    # ------------------------------------------------------------------

    def _load_user_model_and_context(self) -> Tuple[type, Optional[type]]:
        """Dynamically imports the user's SignalModel subclass and Context.

        Returns:
            Tuple of ``(model_class, context_class)``. ``context_class``
            may be ``None`` if the strategy does not define a ``Context``.

        Raises:
            StrategyError: If ``model.py`` is missing or contains no valid
                ``SignalModel`` subclass.
        """
        model_path = os.path.join(self.strategy_dir, "model.py")
        context_path = os.path.join(self.strategy_dir, "context.py")

        if not os.path.exists(model_path):
            raise StrategyError(f"model.py not found in {self.strategy_dir}")

        try:
            # Import context
            context_class = None
            if os.path.exists(context_path):
                ctx_name = f"trainer_ctx_{os.path.basename(self.strategy_dir)}"
                spec_ctx = importlib.util.spec_from_file_location(
                    ctx_name, context_path
                )
                ctx_module = importlib.util.module_from_spec(spec_ctx)
                if spec_ctx and spec_ctx.loader:
                    spec_ctx.loader.exec_module(ctx_module)
                context_class = getattr(ctx_module, "Context", None)

            # Import model
            mod_name = f"trainer_model_{os.path.basename(self.strategy_dir)}"
            spec = importlib.util.spec_from_file_location(mod_name, model_path)
            module = importlib.util.module_from_spec(spec)

            if self.strategy_dir not in sys.path:
                sys.path.insert(0, self.strategy_dir)
            try:
                sys.modules[mod_name] = module
                if spec and spec.loader:
                    spec.loader.exec_module(module)
            finally:
                if self.strategy_dir in sys.path:
                    sys.path.remove(self.strategy_dir)

            from .controller import SignalModel

            for obj_name in dir(module):
                obj = getattr(module, obj_name)
                if (
                    isinstance(obj, type)
                    and issubclass(obj, SignalModel)
                    and obj is not SignalModel
                ):
                    return obj, context_class

            raise StrategyError(
                f"No valid SignalModel subclass found in {model_path}"
            )
        except StrategyError:
            raise
        except Exception as e:
            raise StrategyError(f"Strategy initialization failed: {e}")
