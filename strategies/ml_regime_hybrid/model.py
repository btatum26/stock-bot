"""ML Regime Hybrid Model.

Combines ML-based entry quality scoring with rule-based regime classification
and structural confirmations for long-only swing trading.

Pipeline:
    1. Rule-based regime filter (RISK_OFF / NEUTRAL / RISK_ON)
    2. XGBoost binary classifier (unfavorable / favorable)
    3. Rule-based confirmations (trend, vol, overbought, resistance)
    4. Rule-based exits (regime change, trend break, stops) with ML hysteresis
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight

from context import Context
from engine.core.controller import SignalModel
from regime import classify_regime, RISK_OFF, RISK_ON
from confirmations import check_entry_confirmations, check_exit_conditions

# ---------------------------------------------------------------------------
# SPY market regime guard — module-level cache to avoid redundant fetches
# ---------------------------------------------------------------------------

_SPY_EMA_CACHE: dict = {}


def _fetch_spy_ema(index: pd.DatetimeIndex, ema_period: int = 50):
    """Return (spy_close, spy_ema) arrays aligned to *index*.

    Fetches SPY daily OHLCV via yfinance with a warmup buffer so the EMA is
    fully primed at the start of *index*.  Results are cached per
    (start, end, period) so batch runs across many tickers only hit the
    network once.

    If the fetch fails for any reason both arrays are filled with +inf so
    the guard becomes a no-op (fail-open) rather than silently blocking all
    signals.
    """
    import yfinance as yf

    start_ts = pd.Timestamp(index[0]).tz_localize(None) if index.tz is not None else pd.Timestamp(index[0])
    end_ts   = pd.Timestamp(index[-1]).tz_localize(None) if index.tz is not None else pd.Timestamp(index[-1])

    buffer_days = max(ema_period * 3, 200)
    fetch_start = (start_ts - pd.Timedelta(days=buffer_days)).strftime("%Y-%m-%d")
    fetch_end   = (end_ts   + pd.Timedelta(days=5)).strftime("%Y-%m-%d")

    cache_key = (fetch_start, fetch_end, ema_period)
    if cache_key not in _SPY_EMA_CACHE:
        try:
            raw = yf.download("SPY", start=fetch_start, end=fetch_end,
                              interval="1d", progress=False, auto_adjust=True)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.droplevel(1)
            if raw.empty:
                raise ValueError("Empty SPY response")
            raw.index = pd.to_datetime(raw.index).tz_localize(None)
            spy_close = raw["Close"].astype(float)
            spy_ema   = spy_close.ewm(span=ema_period, adjust=False).mean()
            _SPY_EMA_CACHE[cache_key] = (spy_close, spy_ema)
        except Exception:
            _SPY_EMA_CACHE[cache_key] = None

    cached = _SPY_EMA_CACHE[cache_key]
    n = len(index)
    if cached is None:
        # Fail-open: return +inf so spy_close >= spy_ema is always True
        inf = np.full(n, np.inf)
        return inf, inf

    spy_close_s, spy_ema_s = cached
    idx = index.tz_localize(None) if index.tz is not None else index
    aligned_close = spy_close_s.reindex(idx, method="ffill").to_numpy(dtype=np.float64).copy()
    aligned_ema   = spy_ema_s.reindex(idx,   method="ffill").to_numpy(dtype=np.float64).copy()
    # Any still-NaN slots (before SPY history): treat as above EMA (fail-open)
    nan_mask = np.isnan(aligned_close) | np.isnan(aligned_ema)
    aligned_close[nan_mask] = np.inf
    aligned_ema  [nan_mask] = 0.0
    return aligned_close, aligned_ema


# ---------------------------------------------------------------------------
# Feature column names (must match manifest → context.py)
# ---------------------------------------------------------------------------

# Raw (unnormalised) feature columns
COL_ATR        = "AverageTrueRange_14"
COL_EMA_20     = "MovingAverage_20_EMA"
COL_EMA_50     = "MovingAverage_50_EMA"
COL_EMA_200    = "MovingAverage_200_EMA"
COL_ADX        = "ADX_14"
COL_RSI_14     = "RSI_14"
COL_RSI_7      = "RSI_7"
COL_ROC_20     = "ROC_20"
COL_BB_UPPER   = "BollingerBands_20_2.0_UPPER"
COL_BB_LOWER   = "BollingerBands_20_2.0_LOWER"
COL_BB_WIDTH   = "BollingerBands_20_2.0_WIDTH"
COL_OBV        = "OBV"
COL_OBV_SMA20  = "OBV_SMA_20"

# Normalised / derived features
COL_NORM_EMA_50  = "Norm_MovingAverage_50_EMA"
COL_NORM_SMA_252 = "Norm_MovingAverage_252_SMA"

# S/R levels
COL_SR_RES = "SupportResistance_0.02_Bill Williams_1.0_0.015_35_NEAREST_RESISTANCE_LEVEL"
COL_SR_SUP = "SupportResistance_0.02_Bill Williams_1.0_0.015_35_NEAREST_SUPPORT_LEVEL"

# Mapping passed to regime.classify_regime so it uses feature columns
REGIME_FEATURE_MAP = {
    "atr":        COL_ATR,
    "ema_50":     COL_EMA_50,
    "ema_200":    COL_EMA_200,
    "norm_ema_50": COL_NORM_EMA_50,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _inverse_transform_column(
    scaled_values: np.ndarray,
    scaler,
    feature_cols: list,
    col_name: str,
) -> np.ndarray:
    """Recover raw values for one column via MinMaxScaler inverse."""
    if scaler is None or col_name not in feature_cols:
        return np.full_like(scaled_values, np.nan, dtype=np.float64)
    idx = feature_cols.index(col_name)
    return (scaled_values - scaler.min_[idx]) / scaler.scale_[idx]


def _compute_bb_pct_b(bb_upper: np.ndarray, bb_lower: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Bollinger %B from pre-computed bands."""
    width = bb_upper - bb_lower
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_b = np.where(width > 0, (close - bb_lower) / width, np.nan)
    return pct_b


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MLRegimeHybridModel(SignalModel):
    """Long-only swing strategy: ML regime scoring + rule-based gates."""

    # ---------------------------------------------------------------
    # build_labels  (called once per ticker during training)
    # ---------------------------------------------------------------

    def build_labels(
        self, df: pd.DataFrame, context: Context, params: dict
    ) -> pd.Series:
        """Build binary labels for RISK_ON bars only.

        Classes:
            0 UNFAVORABLE — forward return below threshold
            1 FAVORABLE   — forward return at or above min_profit_threshold

        Regime persistence is deliberately excluded from the label so that
        the model learns return quality, not regime persistence.  The regime
        filter is applied at inference time by the signal state machine.

        Non-RISK_ON bars and the final ``lookforward`` bars → NaN.
        """
        lookforward      = int(params.get("lookforward", 20))
        min_profit       = float(params.get("min_profit_threshold", 0.03))

        regime_now = classify_regime(df, params, feature_cols=REGIME_FEATURE_MAP)

        close = df["close"] if "close" in df.columns else df["Close"]
        future_close = close.shift(-lookforward)
        fwd_return = (future_close - close) / close.abs().replace(0, 1e-9)

        # Default: UNFAVORABLE (0)
        y = pd.Series(0.0, index=df.index)

        # FAVORABLE (1): forward return meets the profit threshold
        good = fwd_return >= min_profit
        y[good] = 1.0

        # Only label RISK_ON bars; final lookforward bars are unknown
        y[regime_now != RISK_ON] = np.nan
        y.iloc[-lookforward:]    = np.nan

        return y

    # ---------------------------------------------------------------
    # fit_model  (called once on pooled X, y)
    # ---------------------------------------------------------------

    @staticmethod
    def _augment_features(X: pd.DataFrame) -> pd.DataFrame:
        """Add derived binary features that encode structural information.

        - ``_no_overhead_resistance``: NaN in SR resistance means price is in
          all-time-high territory — make the semantic explicit as a 0/1 flag.
        - ``_obv_above_sma``: OBV above its 20-bar SMA signals rising volume
          trend (buying pressure accelerating).
        """
        X = X.copy()
        if COL_SR_RES in X.columns:
            X["_no_overhead_resistance"] = X[COL_SR_RES].isna().astype(float)
        if COL_OBV in X.columns and COL_OBV_SMA20 in X.columns:
            X["_obv_above_sma"] = (X[COL_OBV] > X[COL_OBV_SMA20]).astype(float)
        return X

    def fit_model(self, X, y, params: dict) -> dict:
        """Fit XGBoost binary classifier on the pooled (X, y) matrix."""
        X = self._augment_features(X)

        y_int   = y.astype(int)
        weights = compute_sample_weight(class_weight="balanced", y=y_int)

        clf = XGBClassifier(
            n_estimators     = int(params.get("n_estimators", 300)),
            max_depth        = int(params.get("max_depth", 4)),
            learning_rate    = float(params.get("learning_rate", 0.05)),
            min_child_weight = int(params.get("min_child_weight", 5)),
            subsample        = float(params.get("subsample", 0.7)),
            colsample_bytree = float(params.get("colsample_bytree", 0.7)),
            reg_lambda       = float(params.get("reg_lambda", 2.0)),
            reg_alpha        = float(params.get("reg_alpha", 0.1)),
            gamma            = float(params.get("gamma", 0.1)),
            objective        = "binary:logistic",
            eval_metric      = "logloss",
            tree_method      = "hist",
            n_jobs           = -1,
            random_state     = 42,
        )
        clf.fit(X, y_int, sample_weight=weights)
        return {"model": clf}

    # ---------------------------------------------------------------
    # generate_signals  (inference)
    # ---------------------------------------------------------------

    def generate_signals(
        self,
        df: pd.DataFrame,
        context: Context,
        params: dict,
        artifacts: dict,
    ) -> pd.Series:
        """Generate position signals: 0 = flat, 1 = long.

        Feature columns in ``df`` are MinMax-scaled at this point.
        Rule-based logic that needs *raw* indicator values inverse-
        transforms the relevant columns via the saved scaler.
        """
        n            = len(df)
        feature_cols = artifacts.get("feature_cols", [])
        scaler       = artifacts.get("system_scaler")

        # --- 1. Regime ---------------------------------------------------
        # Inverse-transform scaled feature columns back to raw values for
        # the regime classifier (which uses absolute comparisons).
        regime_fc = {}
        if COL_ATR in feature_cols:
            regime_fc["atr"] = COL_ATR  # rank-based; scale-invariant

        df = df.copy()
        if COL_NORM_EMA_50 in feature_cols:
            raw_norm50 = _inverse_transform_column(
                df[COL_NORM_EMA_50].values, scaler, feature_cols, COL_NORM_EMA_50
            )
            df["__raw_norm_ema50"] = raw_norm50
            regime_fc["norm_ema_50"] = "__raw_norm_ema50"

        if COL_EMA_50 in feature_cols and COL_EMA_200 in feature_cols:
            df["__raw_ema50"]  = _inverse_transform_column(
                df[COL_EMA_50].values, scaler, feature_cols, COL_EMA_50
            )
            df["__raw_ema200"] = _inverse_transform_column(
                df[COL_EMA_200].values, scaler, feature_cols, COL_EMA_200
            )
            regime_fc["ema_50"]  = "__raw_ema50"
            regime_fc["ema_200"] = "__raw_ema200"

        regime     = classify_regime(df, params, feature_cols=regime_fc)
        regime_arr = regime.to_numpy()

        # --- 2. ML predictions -------------------------------------------
        clf          = artifacts.get("model")
        p_favorable  = np.zeros(n, dtype=np.float64)

        if clf is not None and feature_cols:
            X = self._augment_features(df[feature_cols])
            p_favorable = clf.predict_proba(X)[:, 1]  # class 1 = favorable

        # --- 3. Inverse-transform for rule confirmations -----------------
        def _raw_col(col_name: str) -> np.ndarray:
            if col_name in feature_cols and scaler is not None:
                return _inverse_transform_column(
                    df[col_name].values, scaler, feature_cols, col_name
                )
            if col_name in df.columns:
                return df[col_name].values.astype(np.float64)
            return np.full(n, np.nan)

        raw_ema50    = _raw_col(COL_EMA_50)
        raw_ema200   = _raw_col(COL_EMA_200)
        raw_adx      = _raw_col(COL_ADX)
        raw_rsi      = _raw_col(COL_RSI_14)
        raw_bb_upper = _raw_col(COL_BB_UPPER)
        raw_bb_lower = _raw_col(COL_BB_LOWER)
        raw_res      = _raw_col(COL_SR_RES)

        # ATR for regime confirmations (percentile + roc)
        raw_atr  = _raw_col(COL_ATR)
        atr_s    = pd.Series(raw_atr, index=df.index)
        atr_pct  = (
            atr_s.rolling(int(params.get("atr_lookback", 252)), min_periods=1)
            .rank(pct=True)
            .to_numpy(dtype=np.float64)
        )
        atr_roc  = (atr_s / atr_s.shift(5) - 1.0).to_numpy(dtype=np.float64)

        # Close — OHLCV columns are raw prices (price_normalization="none")
        close_col = "close" if "close" in df.columns else "Close"
        raw_close = df[close_col].values.astype(np.float64)

        bb_pct_b  = _compute_bb_pct_b(raw_bb_upper, raw_bb_lower, raw_close)

        # --- 3b. SPY market regime guard ---------------------------------
        spy_ema_period = int(params.get("spy_ema_period", 50))
        spy_close_arr, spy_ema50_arr = _fetch_spy_ema(df.index, ema_period=spy_ema_period)
        # True when SPY is above its EMA → entries allowed
        spy_bull = spy_close_arr >= spy_ema50_arr

        # --- 4. Thresholds -----------------------------------------------
        eq_thr    = float(params.get("entry_quality_threshold", 0.60))
        exit_thr  = float(params.get("exit_quality_threshold", 0.30))
        min_hold  = int(params.get("min_hold_days", 10))
        max_hold  = int(params.get("max_hold_days", 200))

        # --- 5. State machine --------------------------------------------
        signals     = np.zeros(n, dtype=np.float64)
        position    = 0.0
        entry_price = 0.0
        peak_close  = 0.0
        entry_idx   = 0

        for i in range(n):
            c = raw_close[i]
            if np.isnan(c):
                continue

            if position == 0.0:
                # === ENTRY ===
                # Gate 0: SPY must be above its EMA-50 (broad market health)
                if not spy_bull[i]:
                    continue

                # Gate 1: market regime must be RISK_ON
                if regime_arr[i] != RISK_ON:
                    continue

                # Gate 2: ML must be confident in a favorable outcome
                if p_favorable[i] < eq_thr:
                    continue

                # Gate 3: rule-based structural confirmations
                conf = check_entry_confirmations(
                    close           = c,
                    ema_50          = raw_ema50[i],
                    ema_200         = raw_ema200[i],
                    adx             = raw_adx[i],
                    atr_pct         = atr_pct[i],
                    atr_roc         = atr_roc[i],
                    rsi             = raw_rsi[i],
                    bb_pct_b        = bb_pct_b[i],
                    nearest_resistance = raw_res[i],
                    params          = params,
                )
                if not conf.passed:
                    continue

                position    = 1.0
                entry_price = c
                peak_close  = c
                entry_idx   = i
                signals[i]  = 1.0

            else:
                # === EXIT ===
                if c > peak_close:
                    peak_close = c

                bars_held = i - entry_idx

                # Hard time-stop (no ML override)
                if bars_held >= max_hold:
                    position = 0.0
                    continue

                # Hard exit: SPY dropped below EMA-50 — exit regardless of
                # hold period or ML confidence to prevent stop cascades.
                if not spy_bull[i]:
                    position = 0.0
                    continue

                should_exit, reason = check_exit_conditions(
                    close            = c,
                    ema_50           = raw_ema50[i],
                    entry_price      = entry_price,
                    peak_since_entry = peak_close,
                    current_regime   = int(regime_arr[i]),
                    params           = params,
                )

                # Hard exits: stop-loss, trailing-stop, regime collapse
                # Execute immediately regardless of hold period or ML confidence.
                if should_exit and reason in ("stop_loss", "trailing_stop", "regime_risk_off"):
                    position = 0.0
                    continue

                # Soft exits: only fire after min_hold AND when ML agrees
                # (p_favorable below exit hysteresis threshold).
                # This prevents whipsawing on brief trend-breaks and
                # prevents exiting high-conviction holds on noise.
                if bars_held >= min_hold:
                    soft_rule_exit = should_exit and reason == "trend_break_ema50"
                    ml_exit        = p_favorable[i] < exit_thr
                    if soft_rule_exit or ml_exit:
                        position = 0.0
                        continue

                signals[i] = 1.0

        return pd.Series(signals, index=df.index)
