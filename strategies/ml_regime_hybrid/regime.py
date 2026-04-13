"""Rule-based regime classification.

Classifies each bar into one of three regimes using volatility and trend
conditions.  Works with both raw-OHLCV DataFrames and DataFrames where
OHLCV has been price-normalised (e.g. log returns) — as long as the
relevant *feature columns* are available in `df`.

When feature columns are present (they are computed from raw prices
before price normalisation), the classifier uses them directly.
When they're absent, it falls back to computing indicators from OHLCV.

Regimes:
    RISK_OFF (0) - High volatility or downtrend.  Stay flat.
    NEUTRAL  (1) - No clear edge.  Hold existing, don't enter.
    RISK_ON  (2) - Low-vol uptrend with stable vol.  Look for entries.
"""

import numpy as np
import pandas as pd

RISK_OFF = 0
NEUTRAL = 1
RISK_ON = 2


# ------------------------------------------------------------------
# Helpers for the fallback path (raw-OHLCV computation)
# ------------------------------------------------------------------

def _ohlcv_col(df: pd.DataFrame, name: str) -> pd.Series:
    return df[name] if name in df.columns else df[name.capitalize()]


def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR from raw OHLCV."""
    high = _ohlcv_col(df, "high")
    low = _ohlcv_col(df, "low")
    close = _ohlcv_col(df, "close")
    prev = close.shift(1)
    tr = np.maximum(high - low, np.maximum((high - prev).abs(), (low - prev).abs()))
    return tr.rolling(window=period, min_periods=period).mean()


def _compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def classify_regime(
    df: pd.DataFrame,
    params: dict,
    feature_cols: dict | None = None,
) -> pd.Series:
    """Classify each bar into RISK_OFF (0), NEUTRAL (1), or RISK_ON (2).

    Args:
        df: DataFrame that *always* has OHLCV columns.  May also
            contain pre-computed feature columns.
        params: Hyperparameters (thresholds, lookbacks).
        feature_cols: Optional mapping of logical names to actual
            DataFrame column names.  Recognised keys:

            * ``atr``  – raw ATR column
            * ``ema_50``  – raw EMA-50 column
            * ``ema_200`` – raw EMA-200 column
            * ``norm_ema_50`` – pct_distance-normalised EMA-50
                (positive ⇒ close > EMA-50)

            When a key is missing the classifier computes the
            indicator from OHLCV (fallback path).
    """
    low_vol_thr = float(params.get("low_vol_threshold", 0.25))
    high_vol_thr = float(params.get("high_vol_threshold", 0.75))
    atr_lookback = int(params.get("atr_lookback", 252))
    vol_limit = float(params.get("vol_expansion_limit", 0.10))
    fc = feature_cols or {}

    # ---------- Volatility ----------
    if "atr" in fc and fc["atr"] in df.columns:
        atr = df[fc["atr"]]
    else:
        atr = _compute_atr(df, period=14)

    atr_pct = atr.rolling(atr_lookback, min_periods=atr_lookback).rank(pct=True)
    atr_roc = atr / atr.shift(5) - 1.0

    low_vol = atr_pct < low_vol_thr
    high_vol = atr_pct > high_vol_thr
    vol_stable = atr_roc < vol_limit

    # ---------- Trend ----------
    # Prefer the pct_distance-normalised EMA-50 feature when available
    # because it remains valid even after log-return price normalisation.
    if "norm_ema_50" in fc and fc["norm_ema_50"] in df.columns:
        # pct_distance = (close - EMA50) / EMA50
        # positive ⇒ close > EMA50
        norm_50 = df[fc["norm_ema_50"]]
        close_above_50 = norm_50 > 0
    elif "ema_50" in fc and fc["ema_50"] in df.columns:
        close = _ohlcv_col(df, "close")
        close_above_50 = close > df[fc["ema_50"]]
    else:
        close = _ohlcv_col(df, "close")
        close_above_50 = close > _compute_ema(close, 50)

    if "ema_50" in fc and fc["ema_50"] in df.columns and "ema_200" in fc and fc["ema_200"] in df.columns:
        ema50_above_200 = df[fc["ema_50"]] > df[fc["ema_200"]]
    elif "ema_50" not in fc and "ema_200" not in fc:
        close = _ohlcv_col(df, "close")
        ema_50 = _compute_ema(close, 50)
        ema_200 = _compute_ema(close, 200)
        ema50_above_200 = ema_50 > ema_200
    else:
        # Partial — fall back to computing both
        close = _ohlcv_col(df, "close")
        ema50_above_200 = _compute_ema(close, 50) > _compute_ema(close, 200)

    uptrend = close_above_50 & ema50_above_200
    downtrend = ~close_above_50 & ~ema50_above_200

    # ---------- Combine ----------
    regime = pd.Series(NEUTRAL, index=df.index, dtype=np.int32)
    regime[low_vol & uptrend & vol_stable] = RISK_ON
    regime[high_vol | downtrend] = RISK_OFF

    return regime
