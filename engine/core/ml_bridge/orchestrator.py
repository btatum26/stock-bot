import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, Any, Tuple, List, Optional, Iterable
from ..logger import logger

# Default FFD parameters (Lopez de Prado fixed-window fractional differentiation).
# d=0.4 preserves memory while producing a stationary series for most equity time
# series; window=10 caps weight accumulation so early rows don't dominate.
DEFAULT_FFD_D = 0.4
DEFAULT_FFD_WINDOW = 10


class MLBridge:
    """
    Acts as the strict pre-processing boundary between Phase 2 features and Phase 3 models.

    This class ensures that Machine Learning feature matrices are correctly purged of
    NaN values caused by rolling indicator windows, and enforces strictly stateful
    dataset scaling to prevent lookahead bias during inference.
    """

    # ------------------------------------------------------------------
    # Fractional differentiation (Lopez de Prado, Ch. 5)
    # ------------------------------------------------------------------

    @staticmethod
    def _ffd_weights(d: float, window: int) -> np.ndarray:
        """Fixed-window FFD weights.

        Returns weights ordered so that ``weights[-1]`` is the current-bar
        weight (``1.0``) and ``weights[0]`` is the oldest. Designed for
        dot-product against a trailing window of raw values.
        """
        w = [1.0]
        for k in range(1, window):
            next_w = -w[-1] * (d - k + 1) / k
            w.append(next_w)
        return np.array(w[::-1], dtype=float)

    @staticmethod
    def apply_ffd_series(
        series: pd.Series,
        d: float = DEFAULT_FFD_D,
        window: int = DEFAULT_FFD_WINDOW,
    ) -> pd.Series:
        """Applies fixed-window FFD to a single series.

        Produces NaN for the first ``window - 1`` values where there is
        insufficient history for a full weighted dot product.
        """
        values = series.to_numpy(dtype=float)
        n = len(values)
        out = np.full(n, np.nan)
        if n < window:
            return pd.Series(out, index=series.index, name=series.name)
        weights = MLBridge._ffd_weights(d, window)
        for i in range(window - 1, n):
            segment = values[i - window + 1: i + 1]
            if np.isnan(segment).any():
                continue
            out[i] = float(np.dot(weights, segment))
        return pd.Series(out, index=series.index, name=series.name)

    @staticmethod
    def apply_ffd_to_dataframe(
        df: pd.DataFrame,
        columns: Iterable[str],
        d: float = DEFAULT_FFD_D,
        window: int = DEFAULT_FFD_WINDOW,
    ) -> pd.DataFrame:
        """Applies FFD in-place to the named columns, then drops the FFD warmup.

        The first ``window - 1`` rows are dropped because FFD cannot produce
        a defined value there. Must be called on a contiguous (per-ticker)
        time-series DataFrame; FFD is path-dependent.
        """
        cols = [c for c in columns if c in df.columns]
        if not cols:
            return df
        out = df.copy()
        for col in cols:
            out[col] = MLBridge.apply_ffd_series(out[col], d=d, window=window)
        if window > 1:
            out = out.iloc[window - 1:]
        return out

    @staticmethod
    def prepare_training_matrix(
        df: pd.DataFrame,
        feature_cols: List[str],
        l_max: int
    ) -> Tuple[pd.DataFrame, Any]:
        """
        Prepares the historical feature matrix for model training.

        This method performs two critical operations:
        1. Purges the first `l_max` rows where rolling features are incomplete (NaN).
        2. Fits a global MinMaxScaler strictly on the defined feature columns, bounding 
           the data between [-1.0, 1.0].

        Args:
            df (pd.DataFrame): The raw historical dataset outputted by the FeatureOrchestrator.
            feature_cols (List[str]): A definitive list of column names that will be fed 
                into the model as features (X).
            l_max (int): The maximum lookback window discovered across all computed 
                features (e.g., if a 200-SMA is used, l_max is 200).

        Returns:
            Tuple[pd.DataFrame, Any]: 
                - df_clean (pd.DataFrame): The truncated and scaled dataset ready for `SignalModel.train()`.
                - scaler (Any): The fitted scikit-learn scaler. This MUST be saved via the ArtifactManager.

        Raises:
            ValueError: If `feature_cols` contains columns not present in the DataFrame.
            ValueError: If `l_max` is greater than or equal to the length of the DataFrame.

        Example:
            >>> df_train, scaler = MLBridge.prepare_training_matrix(df, ['RSI_14', 'MACD'], 200)
            >>> artifacts['system_scaler'] = scaler
        """
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing feature columns in DataFrame: {missing_cols}")
            
        if l_max >= len(df):
            raise ValueError(f"l_max ({l_max}) is larger than the dataset length ({len(df)}).")

        # 1. NaN Purge: Drop the top l_max rows safely
        df_clean = df.iloc[l_max:].copy()

        # 2. Stateful Scaling: Fit and transform
        scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
        df_clean[feature_cols] = scaler.fit_transform(df_clean[feature_cols])

        logger.info(f"Training matrix prepared. Purged {l_max} rows. Scaler fitted.")
        return df_clean, scaler

    @staticmethod
    def collect_non_stationary_columns(
        features_config: List[Dict[str, Any]],
    ) -> List[str]:
        """Asks each configured feature which of its outputs are non-stationary.

        Returns a deduplicated list of column names that should be routed
        through FFD before scaling. Feature classes own this knowledge via
        ``Feature.non_stationary_outputs(params)``; unknown IDs are silently
        skipped (caller already validates registry membership).
        """
        from ..features.base import FEATURE_REGISTRY

        seen: List[str] = []
        for cfg in features_config:
            fid = cfg.get("id")
            params = cfg.get("params", {}) or {}
            feature_cls = FEATURE_REGISTRY.get(fid)
            if feature_cls is None:
                continue
            try:
                cols = feature_cls().non_stationary_outputs(params) or []
            except Exception as e:
                logger.warning(
                    f"non_stationary_outputs failed for {fid}: {e}. "
                    f"Treating as stationary."
                )
                cols = []
            for c in cols:
                if c not in seen:
                    seen.append(c)
        return seen

    @staticmethod
    def prepare_inference_matrix(
        df: pd.DataFrame,
        feature_cols: List[str],
        l_max: int,
        artifacts: Dict[str, Any],
        is_live: bool = False
    ) -> pd.DataFrame:
        """
        Prepares the feature matrix for backtesting or live execution.

        Crucially, this method NEVER fits a new scaler. It transforms the incoming 
        data using the exact scaler fitted during `ExecutionMode.TRAIN`. If no scaler 
        is found in the artifacts (e.g., a rule-based strategy), it skips scaling.

        Args:
            df (pd.DataFrame): The dataset outputted by the FeatureOrchestrator.
            feature_cols (List[str]): The exact list of feature columns used during training.
            l_max (int): The maximum lookback window required to compute current features.
            artifacts (Dict[str, Any]): The loaded artifact dictionary.
            is_live (bool, optional): If True, aggressively truncates the DataFrame to 
                only the single most recent row to optimize live execution latency. 
                Defaults to False (Backtest mode).

        Returns:
            pd.DataFrame: The pre-processed dataframe ready to be passed into the 
                user's `generate_signals()` method.

        Raises:
            KeyError: If a scaler is expected but not found in the artifacts.
        """
        # 1. Row Selection & Purging
        if is_live:
            # For live execution, we only need the absolute latest row (after ensuring features computed)
            df_clean = df.iloc[[-1]].copy()
        else:
            # For backtesting, we purge the top l_max rows just like in training
            df_clean = df.iloc[l_max:].copy()

        # 2. Replay FFD on non-stationary feature columns using the config
        #    persisted at training time. Must happen before scaling so the
        #    distribution the scaler sees matches the one it was fit on.
        ffd_cols = artifacts.get("ffd_columns") or []
        if ffd_cols:
            ffd_d = float(artifacts.get("ffd_d", DEFAULT_FFD_D))
            ffd_window = int(artifacts.get("ffd_window", DEFAULT_FFD_WINDOW))
            df_clean = MLBridge.apply_ffd_to_dataframe(
                df_clean, ffd_cols, d=ffd_d, window=ffd_window
            )

        # 3. Extract and Apply Scaler
        scaler = artifacts.get("system_scaler")

        if scaler:
            # Strict TRANSFORM only. Never fit.
            df_clean[feature_cols] = scaler.transform(df_clean[feature_cols])
            logger.debug("Applied loaded scaler to inference matrix.")
        else:
            logger.debug("No system_scaler found in artifacts. Passing unscaled data.")

        return df_clean

    @staticmethod
    def apply_price_normalization(
        df: pd.DataFrame,
        method: str,
        ffd_d: float = DEFAULT_FFD_D,
        ffd_window: int = DEFAULT_FFD_WINDOW,
    ) -> pd.DataFrame:
        """Normalizes raw OHLCV price columns before model training or inference.

        Applied after feature computation and warmup purge so computed features
        (which depend on raw prices) are unaffected.

        Args:
            df: DataFrame containing OHLCV columns (lowercase names).
            method: Normalization method — ``'log_returns'``, ``'ffd'``, or ``'none'``.
            ffd_d: Fractional differentiation order when ``method='ffd'``.
            ffd_window: Fixed-window size for FFD when ``method='ffd'``.

        Returns:
            DataFrame with normalized price columns. ``'log_returns'`` drops the
            first row (NaN from shift). ``'ffd'`` drops the first ``window-1``
            rows (NaN from the fixed-window weighted sum).
        """
        if not method or method == "none":
            return df

        df = df.copy()

        if method == "log_returns":
            for col in ("open", "high", "low", "close"):
                if col in df.columns:
                    df[col] = np.log(df[col] / df[col].shift(1))
            if "volume" in df.columns:
                # Volume is already mean-reverting; use log-level rather than log-diff
                df["volume"] = np.log(df["volume"].replace(0, np.nan)).fillna(0)
            df = df.iloc[1:]  # drop NaN row introduced by shift on price columns
        elif method == "ffd":
            # Fractional differentiation on log prices: makes OHLC stationary
            # while preserving long-memory information the model needs.
            # Volume is log-level (already mean-reverting, same as log_returns).
            price_cols = [c for c in ("open", "high", "low", "close") if c in df.columns]
            for col in price_cols:
                df[col] = np.log(df[col].replace(0, np.nan))
            df = MLBridge.apply_ffd_to_dataframe(
                df, price_cols, d=ffd_d, window=ffd_window
            )
            if "volume" in df.columns:
                df["volume"] = np.log(df["volume"].replace(0, np.nan)).fillna(0)
        else:
            logger.warning(
                f"Unknown price normalization method '{method}'. Skipping."
            )

        return df