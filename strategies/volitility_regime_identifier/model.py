import numpy as np
import pandas as pd
from context import Context
from engine.core.controller import SignalModel


class VolatilityRegimeModel(SignalModel):
    """
    Volatility Regime Model V1
    ==========================
    Long-only swing strategy. Trades trend-following setups only when realized
    volatility sits in its low-regime bucket; stays flat otherwise.

    Regimes (ATR-14 percentile over `atr_lookback`):
        LOW_VOL   <  low_vol_threshold    -> eligible to enter long
        NORMAL    in [low, high]          -> hold if already long, do not enter
        HIGH_VOL  >= high_vol_threshold   -> force flat

    Entry (all must be true):
        regime == LOW_VOL
        Close > EMA_fast
        EMA_fast > EMA_slow
        ATR_ROC(5) < atr_roc_max     (vol not expanding sharply)

    Exit (any triggers flat):
        regime == HIGH_VOL
        Close < EMA_fast              (trend break)
        (Close - entry) / entry <= -stop_loss

    ATR-percentile and ATR-ROC are computed inside this method because the
    AverageTrueRange feature only ships with none/z_score/pct_distance/price_ratio
    normalizers.
    """

    def train(self, df: pd.DataFrame, context: Context, params: dict) -> dict:
        return {}

    def generate_signals(
        self,
        df: pd.DataFrame,
        context: Context,
        params: dict,
        artifacts: dict,
    ) -> pd.Series:
        f = context.features
        p = context.params

        low_thr       = float(params.get("low_vol_threshold",  p.low_vol_threshold))
        high_thr      = float(params.get("high_vol_threshold", p.high_vol_threshold))
        atr_lookback  = int(params.get("atr_lookback",         p.atr_lookback))
        atr_roc_max   = float(params.get("atr_roc_max",        p.atr_roc_max))
        stop_loss     = float(params.get("stop_loss",          p.stop_loss))

        atr       = df[f.AverageTrueRange_14]
        ema_fast  = df[f.MovingAverage_50_EMA]
        ema_slow  = df[f.MovingAverage_200_EMA]
        close     = df['close'] if 'close' in df.columns else df['Close']

        # ATR percentile rank over the rolling window: fraction of window values
        # the current ATR is >=. rank(pct=True) on the last element of each window.
        atr_pct = atr.rolling(atr_lookback, min_periods=atr_lookback).apply(
            lambda w: (w <= w[-1]).mean(), raw=True
        )

        atr_roc_5 = atr / atr.shift(5) - 1.0

        low_vol   = atr_pct < low_thr
        high_vol  = atr_pct >= high_thr

        uptrend   = (close > ema_fast) & (ema_fast > ema_slow)
        vol_stable = atr_roc_5 < atr_roc_max

        entry_ok = low_vol & uptrend & vol_stable

        # State machine: stop-loss needs entry price, so iterate once.
        close_arr     = close.to_numpy()
        entry_arr     = entry_ok.fillna(False).to_numpy()
        high_vol_arr  = high_vol.fillna(False).to_numpy()
        close_below_fast = (close < ema_fast).fillna(False).to_numpy()

        n = len(df)
        out = np.zeros(n, dtype=np.float64)

        in_pos = False
        entry_price = 0.0
        for i in range(n):
            if in_pos:
                c = close_arr[i]
                hit_stop = (c - entry_price) / entry_price <= -stop_loss
                if high_vol_arr[i] or close_below_fast[i] or hit_stop:
                    in_pos = False
                    entry_price = 0.0
                    # out[i] stays 0.0 — exit this bar
                else:
                    out[i] = 1.0
            else:
                if entry_arr[i]:
                    in_pos = True
                    entry_price = close_arr[i]
                    out[i] = 1.0

        return pd.Series(out, index=df.index)
