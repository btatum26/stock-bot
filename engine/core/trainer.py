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
        # FFD parameters applied to features self-declared as non-stationary
        # and to prices when price_normalization == "ffd". See MLBridge for
        # the algorithm (Lopez de Prado fixed-window).
        "ffd_d": 0.4,
        "ffd_window": 10,
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
            # TODO: improve logging — add structured exc_info so the full traceback is visible
            raise StrategyError(
                f"Missing or invalid manifest in {self.strategy_dir}: {e}"
            ) from e

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

        ffd_d = float(self.training_config.get("ffd_d", 0.4))
        ffd_window = int(self.training_config.get("ffd_window", 10))

        # Apply price normalization before the model sees OHLCV columns.
        # Must happen after feature computation (features use raw prices) and
        # after warmup purge (so log-diff NaN drop only costs 1 row).
        price_norm = self.training_config.get("price_normalization", "none")
        if price_norm != "none":
            df_clean = MLBridge.apply_price_normalization(
                df_clean, price_norm, ffd_d=ffd_d, ffd_window=ffd_window
            )
            logger.info(f"Price normalization applied: {price_norm}")

        # Apply FFD to features self-declared as non-stationary. Must happen on
        # the contiguous df_clean (FFD is path-dependent) before any splitting
        # or scaling. Features that inherit the default [] are untouched.
        ffd_columns = MLBridge.collect_non_stationary_columns(features_config)
        ffd_columns = [c for c in ffd_columns if c in df_clean.columns]
        if ffd_columns:
            df_clean = MLBridge.apply_ffd_to_dataframe(
                df_clean, ffd_columns, d=ffd_d, window=ffd_window
            )
            logger.info(
                f"FFD applied to {len(ffd_columns)} non-stationary feature "
                f"columns (d={ffd_d}, window={ffd_window})."
            )

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

        # Persist FFD metadata so prepare_inference_matrix can replay the
        # identical transform when the backtester or signal path loads the
        # artifacts back.
        if ffd_columns:
            artifacts["ffd_columns"] = ffd_columns
            artifacts["ffd_d"] = ffd_d
            artifacts["ffd_window"] = ffd_window

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

    @staticmethod
    def _compute_fold_diagnostics(
        train_metrics_list: List[Dict[str, Any]],
        val_metrics_list: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compute fold-level Sharpe distribution and IS/OOS rank correlation.

        Returns a dict with:
          fold_sharpes            : OOS Sharpe per fold (raw list)
          fold_train_sharpes      : IS  Sharpe per fold (raw list)
          fraction_positive_folds : fraction of OOS folds with Sharpe > 0
          fraction_above_half_folds : fraction with Sharpe > 0.5
          spearman_is_oos         : Spearman rank corr between IS and OOS Sharpes
        """
        from scipy.stats import spearmanr

        def _safe_sharpe(m: Dict[str, Any]) -> float:
            v = m.get("Sharpe Ratio")
            if isinstance(v, (int, float)) and math.isfinite(v):
                return float(v)
            return float("nan")

        val_sharpes   = [_safe_sharpe(m) for m in val_metrics_list]
        train_sharpes = [_safe_sharpe(m) for m in train_metrics_list]

        finite_val = [s for s in val_sharpes if math.isfinite(s)]
        n = len(finite_val)
        if n == 0:
            return {}

        fraction_positive  = sum(1 for s in finite_val if s > 0.0) / n
        fraction_above_half = sum(1 for s in finite_val if s > 0.5) / n

        spearman = float("nan")
        finite_pairs = [
            (t, v) for t, v in zip(train_sharpes, val_sharpes)
            if math.isfinite(t) and math.isfinite(v)
        ]
        if len(finite_pairs) >= 3:
            try:
                r, _ = spearmanr(
                    [p[0] for p in finite_pairs],
                    [p[1] for p in finite_pairs],
                )
                spearman = float(r)
            except Exception:
                pass

        return {
            "fold_sharpes":             val_sharpes,
            "fold_train_sharpes":       train_sharpes,
            "fraction_positive_folds":  round(fraction_positive, 4),
            "fraction_above_half_folds": round(fraction_above_half, 4),
            "spearman_is_oos":          round(spearman, 4) if math.isfinite(spearman) else float("nan"),
        }

    def _save_diagnostics(self, diagnostics: Dict[str, Any]) -> None:
        """Persist fold diagnostics to strategies/<name>/diagnostics.json."""
        from datetime import datetime, timezone
        path = os.path.join(self.strategy_dir, "diagnostics.json")
        try:
            data = {**diagnostics, "last_trained": datetime.now(timezone.utc).isoformat()}
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save diagnostics.json: {e}")

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
            fold_diagnostics = self._compute_fold_diagnostics(
                [r["train_metrics"] for r in fold_results],
                [r["val_metrics"]   for r in fold_results],
            )
            if fold_diagnostics:
                self._save_diagnostics(fold_diagnostics)
            self._print_training_summary(train_scalar, val_scalar, split_info)
            return {
                "train_metrics":    train_scalar,
                "val_metrics":      val_scalar,
                "split_info":       split_info,
                "params":           hyperparams,
                "fold_diagnostics": fold_diagnostics,
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
            if "n_tickers" in split_info:
                print(f"  Tickers:      {split_info['n_tickers']}")
                print(f"  Total Rows:   {split_info['n_rows']}")

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
        ffd_d = float(self.training_config.get("ffd_d", 0.4))
        ffd_window = int(self.training_config.get("ffd_window", 10))

        # Compute FFD column list once — it depends only on the manifest's
        # features config, not per-ticker data.
        ffd_columns_cfg = MLBridge.collect_non_stationary_columns(features_config)

        # 1. Per-ticker prep
        prepared: Dict[str, Dict[str, Any]] = {}
        feature_cols: Optional[List[str]] = None
        applied_ffd_cols: List[str] = []
        for ticker, raw in datasets.items():
            df_full, l_max = compute_all_features(raw, features_config)
            df_clean = df_full.iloc[l_max:].copy()
            if price_norm != "none":
                df_clean = MLBridge.apply_price_normalization(
                    df_clean, price_norm, ffd_d=ffd_d, ffd_window=ffd_window
                )
            # FFD must be applied per-ticker on the contiguous df (path-dependent).
            # Skip cols that don't exist in this ticker's df (robust to config drift).
            ticker_ffd_cols = [c for c in ffd_columns_cfg if c in df_clean.columns]
            if ticker_ffd_cols:
                df_clean = MLBridge.apply_ffd_to_dataframe(
                    df_clean, ticker_ffd_cols, d=ffd_d, window=ffd_window
                )
                if not applied_ffd_cols:
                    applied_ffd_cols = ticker_ffd_cols
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

        if applied_ffd_cols:
            logger.info(
                f"FFD applied to {len(applied_ffd_cols)} non-stationary feature "
                f"columns per ticker (d={ffd_d}, window={ffd_window})."
            )

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

        # 3. Generate CPCV date-group folds
        date_folds, all_dates, n_groups = self._split_cpcv_by_date(prepared)

        # 4. Train and evaluate each fold
        fold_results: List[Dict[str, Any]] = []
        for fold_idx, (train_dates, val_dates) in enumerate(date_folds):
            logger.info(
                f"Fold {fold_idx + 1}/{len(date_folds)}: "
                f"train={len(train_dates)} dates, val={len(val_dates)} dates"
            )
            result = self._train_fold_multi(
                model_class=model_class,
                context_class=context_class,
                prepared=prepared,
                feature_cols=feature_cols,
                hyperparams=hyperparams,
                train_dates=train_dates,
                val_dates=val_dates,
            )
            fold_results.append(result)

        # 5. Retrain on full pooled dataset for final artifacts
        logger.info("CPCV validation complete. Retraining on full pooled dataset.")
        final = self._train_final_multi(
            model_class, context_class, prepared, feature_cols, hyperparams
        )
        artifacts = final["artifacts"]

        # Persist FFD metadata so the backtester can replay the identical
        # non-stationary-column transform.
        if applied_ffd_cols:
            artifacts["ffd_columns"] = applied_ffd_cols
            artifacts["ffd_d"] = ffd_d
            artifacts["ffd_window"] = ffd_window

        # 6. Persist
        ArtifactManager.save_artifacts(self.strategy_dir, artifacts)

        # 7. Build report
        k_test = self.training_config["k_test_groups"]
        results = self._build_multi_results(
            fold_results, date_folds, prepared, hyperparams,
            n_groups=n_groups, k_test_groups=k_test,
        )
        feature_analysis = artifacts.get("feature_analysis")
        if feature_analysis:
            self._print_feature_analysis(feature_analysis)
            results["feature_analysis"] = feature_analysis
        return results

    def _split_cpcv_by_date(
        self,
        prepared: Dict[str, Dict[str, Any]],
    ) -> Tuple[List[Tuple[set, set]], pd.DatetimeIndex, int]:
        """CPCV splits across the union of all ticker dates.

        Partitions the sorted union of dates into N contiguous groups,
        then generates all C(N, k) combinations where k groups form the
        test set. Applies embargo by removing a percentage of date-groups
        adjacent to each test block from training.

        Returns:
            Tuple of (folds, all_dates, n_groups) where each fold is a
            ``(train_dates_set, val_dates_set)`` pair.
        """
        import itertools

        all_dates = pd.DatetimeIndex([])
        for p in prepared.values():
            idx = pd.DatetimeIndex(p["df_clean"].index)
            all_dates = all_dates.union(idx)
        all_dates = all_dates.sort_values()

        n_groups = self.training_config["n_groups"]
        k_test = self.training_config["k_test_groups"]
        embargo_pct = self.training_config.get("embargo_pct", 0.01)

        if n_groups < 2:
            raise StrategyError(f"n_groups must be >= 2, got {n_groups}")
        if k_test < 1 or k_test >= n_groups:
            raise StrategyError(
                f"k_test_groups must be in [1, {n_groups - 1}], got {k_test}"
            )

        n_dates = len(all_dates)
        if n_dates < n_groups:
            raise StrategyError(
                f"Not enough dates ({n_dates}) for {n_groups} groups"
            )

        # Partition dates into N contiguous groups
        date_groups = np.array_split(np.arange(n_dates), n_groups)
        embargo_size = max(1, int(n_dates * embargo_pct))

        n_folds = len(list(itertools.combinations(range(n_groups), k_test)))
        logger.info(
            f"CPCV (multi-ticker): {n_groups} groups, k={k_test} -> "
            f"{n_folds} folds (embargo={embargo_pct:.1%})"
        )

        folds: List[Tuple[set, set]] = []
        for test_group_ids in itertools.combinations(range(n_groups), k_test):
            test_indices = np.concatenate([date_groups[g] for g in test_group_ids])
            train_group_ids = [g for g in range(n_groups) if g not in test_group_ids]
            train_indices = np.concatenate([date_groups[g] for g in train_group_ids])

            # Apply embargo: remove train dates near test block boundaries
            if embargo_pct > 0:
                test_set = set(test_indices)
                # Find contiguous test blocks
                sorted_test = np.sort(test_indices)
                breaks = np.where(np.diff(sorted_test) > 1)[0]
                starts = np.concatenate([[0], breaks + 1])
                ends = np.concatenate([breaks, [len(sorted_test) - 1]])
                blocks = [
                    (int(sorted_test[s]), int(sorted_test[e]))
                    for s, e in zip(starts, ends)
                ]
                embargo_mask = np.zeros(len(train_indices), dtype=bool)
                for _, block_end in blocks:
                    embargo_mask |= (train_indices > block_end) & (
                        train_indices <= block_end + embargo_size
                    )
                train_indices = train_indices[~embargo_mask]

            train_dates = set(all_dates[train_indices])
            val_dates = set(all_dates[test_indices])
            folds.append((train_dates, val_dates))

        if not folds:
            raise StrategyError("CPCV splitter produced zero folds")
        return folds, all_dates, n_groups

    def _train_fold_multi(
        self,
        model_class: type,
        context_class: Optional[type],
        prepared: Dict[str, Dict[str, Any]],
        feature_cols: List[str],
        hyperparams: Dict[str, Any],
        train_dates: set,
        val_dates: set,
    ) -> Dict[str, Any]:
        """Trains on one CPCV fold.

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
            train_mask = df.index.isin(train_dates)
            val_mask = df.index.isin(val_dates)
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
            date_range = (
                f"{min(train_dates)} -> {max(train_dates)}"
                if train_dates else "no train dates"
            )
            raise StrategyError(
                f"Fold produced no training data [{date_range}]"
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
        folds: List[Tuple[set, set]],
        prepared: Dict[str, Dict[str, Any]],
        hyperparams: Dict[str, Any],
        n_groups: int = 6,
        k_test_groups: int = 2,
    ) -> Dict[str, Any]:
        """Aggregates CPCV fold results into the final report."""
        skip_keys = {"equity_curve", "portfolio", "bh_portfolio", "trade_log"}

        train_scalar = self._aggregate_fold_metrics(
            [r["train_metrics"] for r in fold_results], skip_keys
        )
        val_scalar = self._aggregate_fold_metrics(
            [r["val_metrics"] for r in fold_results], skip_keys
        )

        n_rows = sum(len(p["df_clean"]) for p in prepared.values())
        split_info = {
            "method": "cpcv",
            "n_folds": len(folds),
            "n_groups": n_groups,
            "k_test_groups": k_test_groups,
            "n_tickers": len(prepared),
            "n_rows": n_rows,
            "tickers": list(prepared.keys()),
        }

        fold_diagnostics = self._compute_fold_diagnostics(
            [r["train_metrics"] for r in fold_results],
            [r["val_metrics"]   for r in fold_results],
        )
        if fold_diagnostics:
            self._save_diagnostics(fold_diagnostics)

        self._print_training_summary(train_scalar, val_scalar, split_info)

        return {
            "train_metrics":    train_scalar,
            "val_metrics":      val_scalar,
            "split_info":       split_info,
            "params":           hyperparams,
            "fold_diagnostics": fold_diagnostics,
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
            # TODO: improve logging — add structured exc_info so the full traceback is visible
            raise StrategyError(f"Strategy initialization failed: {e}") from e
