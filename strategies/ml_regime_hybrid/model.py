"""ML Regime Hybrid Model.

Combines ML-based entry quality scoring with rule-based regime classification
and structural confirmations for long-only swing trading.

Pipeline:
    1. Rule-based regime filter (RISK_OFF / NEUTRAL / RISK_ON)
    2. XGBoost binary classifier (unfavorable / favorable)
    3. Rule-based confirmations (trend, vol, overbought, resistance)
    4. Rule-based exits (regime change, trend break, stops)
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
# Feature column names (must match manifest → context.py)
# ---------------------------------------------------------------------------

# Raw (unnormalised) feature columns
COL_ATR = "AverageTrueRange_14"
COL_EMA_20 = "MovingAverage_20_EMA"
COL_EMA_50 = "MovingAverage_50_EMA"
COL_EMA_200 = "MovingAverage_200_EMA"
COL_ADX = "ADX_14"
COL_RSI = "RSI_14"
COL_BB_UPPER = "BollingerBands_20_2.0_UPPER"
COL_BB_LOWER = "BollingerBands_20_2.0_LOWER"
COL_BB_WIDTH = "BollingerBands_20_2.0_WIDTH"

# Normalised features
COL_NORM_EMA_50 = "Norm_MovingAverage_50_EMA"

# S/R levels
COL_SR_RES = "SupportResistance_0.02_Bill Williams_1.0_0.015_35_NEAREST_RESISTANCE_LEVEL"
COL_SR_SUP = "SupportResistance_0.02_Bill Williams_1.0_0.015_35_NEAREST_SUPPORT_LEVEL"


# Mapping passed to regime.classify_regime so it uses feature columns
# (which contain values computed from *raw* prices) instead of OHLCV.
REGIME_FEATURE_MAP = {
    "atr": COL_ATR,
    "ema_50": COL_EMA_50,
    "ema_200": COL_EMA_200,
    "norm_ema_50": COL_NORM_EMA_50,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe(arr, i):
    """Retrieve value at index, returning NaN for out-of-bounds."""
    v = arr[i]
    return v if np.isfinite(v) else np.nan


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
# ML Output
# ---------------------------------------------------------------------------

class MLOutput:
    __slots__ = ("p_unfavorable", "p_favorable")

    def __init__(self, p0: float, p1: float):
        self.p_unfavorable = p0
        self.p_favorable = p1

    @property
    def entry_quality(self) -> float:
        return self.p_favorable

    @property
    def suggested_action(self) -> str:
        if self.p_favorable > 0.50:
            return "LOOK_FOR_ENTRY"
        return "AVOID"


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
            0 UNFAVORABLE — net forward return negative or regime breaks
            1 FAVORABLE   — positive net forward return, regime persists

        Non-RISK_ON bars and the final ``lookforward`` bars → NaN.
        """
        lookforward = int(params.get("lookforward", 20))

        regime_now = classify_regime(df, params, feature_cols=REGIME_FEATURE_MAP)
        regime_future = regime_now.shift(-lookforward)

        close = df["close"] if "close" in df.columns else df["Close"]
        future_close = close.shift(-lookforward)
        fwd_return = (future_close - close) / close.abs().replace(0, 1e-9)

        # Default: UNFAVORABLE (0)
        y = pd.Series(0.0, index=df.index)

        # FAVORABLE (1): regime persists AND positive forward return
        good = (regime_now == regime_future) & (fwd_return > 0)
        y[good] = 1.0

        # Only label RISK_ON bars
        y[regime_now != RISK_ON] = np.nan
        y.iloc[-lookforward:] = np.nan

        return y

    # ---------------------------------------------------------------
    # fit_model  (called once on pooled X, y)
    # ---------------------------------------------------------------

    def fit_model(self, X, y, params: dict) -> dict:
        """Fit XGBoost binary classifier on the pooled (X, y) matrix."""
        y_int = y.astype(int)
        weights = compute_sample_weight(class_weight="balanced", y=y_int)

        clf = XGBClassifier(
            n_estimators=int(params.get("n_estimators", 300)),
            max_depth=int(params.get("max_depth", 4)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            min_child_weight=int(params.get("min_child_weight", 5)),
            subsample=float(params.get("subsample", 0.7)),
            colsample_bytree=float(params.get("colsample_bytree", 0.7)),
            reg_lambda=float(params.get("reg_lambda", 2.0)),
            reg_alpha=float(params.get("reg_alpha", 0.1)),
            gamma=float(params.get("gamma", 0.1)),
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
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
        n = len(df)
        feature_cols = artifacts.get("feature_cols", [])
        scaler = artifacts.get("system_scaler")

        # --- 1. Regime (from feature columns) -------------------------
        # The regime classifier uses unscaled feature columns when
        # available.  Here features ARE scaled, so we inverse-transform
        # the ones needed for regime classification.
        #
        # ATR: monotonic transform → rank-based percentile unaffected.
        #      We can pass the scaled column directly (ranking is
        #      order-preserving under any monotonic transform).
        # EMA trend: we use Norm_EMA_50 (pct_distance).  After
        #      MinMaxScaling the *sign* can shift.  Inverse-transform it
        #      to recover the original pct_distance.
        regime_fc = {}
        if COL_ATR in feature_cols:
            regime_fc["atr"] = COL_ATR  # rank is scale-invariant
        if COL_NORM_EMA_50 in feature_cols:
            # inverse-transform to recover sign
            raw_norm50 = _inverse_transform_column(
                df[COL_NORM_EMA_50].values, scaler, feature_cols, COL_NORM_EMA_50
            )
            df = df.copy()
            _tmp_col = "__raw_norm_ema50"
            df[_tmp_col] = raw_norm50
            regime_fc["norm_ema_50"] = _tmp_col
        if COL_EMA_50 in feature_cols and COL_EMA_200 in feature_cols:
            # inverse-transform both to compare magnitudes
            raw_50 = _inverse_transform_column(
                df[COL_EMA_50].values, scaler, feature_cols, COL_EMA_50
            )
            raw_200 = _inverse_transform_column(
                df[COL_EMA_200].values, scaler, feature_cols, COL_EMA_200
            )
            _tmp50 = "__raw_ema50"
            _tmp200 = "__raw_ema200"
            df[_tmp50] = raw_50
            df[_tmp200] = raw_200
            regime_fc["ema_50"] = _tmp50
            regime_fc["ema_200"] = _tmp200

        regime = classify_regime(df, params, feature_cols=regime_fc)

        # --- 2. ML predictions (on scaled features) -------------------
        clf = artifacts.get("model")
        # p_favorable: probability of class 1 (favorable)
        p_favorable = np.zeros(n, dtype=np.float64)

        if clf is not None and feature_cols:
            X = df[feature_cols]
            # XGBoost handles NaN natively — pass all rows directly
            raw_proba = clf.predict_proba(X)
            p_favorable = raw_proba[:, 1]  # class 1 = favorable

        # --- 3. Inverse-transform columns for rule confirmations ------
        def _raw_col(col_name: str) -> np.ndarray:
            if col_name in feature_cols and scaler is not None:
                return _inverse_transform_column(
                    df[col_name].values, scaler, feature_cols, col_name
                )
            if col_name in df.columns:
                return df[col_name].values.astype(np.float64)
            return np.full(n, np.nan)

        raw_ema50 = _raw_col(COL_EMA_50)
        raw_ema200 = _raw_col(COL_EMA_200)
        raw_adx = _raw_col(COL_ADX)
        raw_rsi = _raw_col(COL_RSI)
        raw_bb_upper = _raw_col(COL_BB_UPPER)
        raw_bb_lower = _raw_col(COL_BB_LOWER)
        raw_res = _raw_col(COL_SR_RES)

        # ATR for regime confirmations (percentile + roc)
        raw_atr = _raw_col(COL_ATR)
        atr_s = pd.Series(raw_atr, index=df.index)
        atr_pct = (
            atr_s.rolling(int(params.get("atr_lookback", 252)), min_periods=1)
            .rank(pct=True)
            .to_numpy(dtype=np.float64)
        )
        atr_roc = (atr_s / atr_s.shift(5) - 1.0).to_numpy(dtype=np.float64)

        # Close — OHLCV columns are raw prices (price_normalization="none")
        close_col = "close" if "close" in df.columns else "Close"
        raw_close = df[close_col].values.astype(np.float64)

        # BB %B from raw bands
        bb_pct_b = _compute_bb_pct_b(raw_bb_upper, raw_bb_lower, raw_close)

        regime_arr = regime.to_numpy()

        # --- 4. Thresholds ---
        eq_thr = float(params.get("entry_quality_threshold", 0.35))
        max_hold = int(params.get("max_hold_days", 60))

        # --- 5. State machine ---
        signals = np.zeros(n, dtype=np.float64)
        position = 0.0
        entry_price = 0.0
        peak_close = 0.0
        entry_idx = 0

        for i in range(n):
            c = raw_close[i]
            if np.isnan(c):
                continue

            if position == 0.0:
                # === ENTRY ===
                if regime_arr[i] != RISK_ON:
                    continue

                ml = MLOutput(1.0 - p_favorable[i], p_favorable[i])
                if ml.suggested_action != "LOOK_FOR_ENTRY":
                    continue
                if ml.entry_quality < eq_thr:
                    continue

                conf = check_entry_confirmations(
                    close=c,
                    ema_50=raw_ema50[i],
                    ema_200=raw_ema200[i],
                    adx=raw_adx[i],
                    atr_pct=atr_pct[i],
                    atr_roc=atr_roc[i],
                    rsi=raw_rsi[i],
                    bb_pct_b=bb_pct_b[i],
                    nearest_resistance=raw_res[i],
                    params=params,
                )
                if not conf.passed:
                    continue

                position = 1.0
                entry_price = c
                peak_close = c
                entry_idx = i
                signals[i] = 1.0

            else:
                # === EXIT ===
                if c > peak_close:
                    peak_close = c

                if (i - entry_idx) > max_hold:
                    position = 0.0
                    continue

                should_exit, _ = check_exit_conditions(
                    close=c,
                    ema_50=raw_ema50[i],
                    entry_price=entry_price,
                    peak_since_entry=peak_close,
                    current_regime=int(regime_arr[i]),
                    params=params,
                )
                if should_exit:
                    position = 0.0
                else:
                    signals[i] = 1.0

        return pd.Series(signals, index=df.index)
