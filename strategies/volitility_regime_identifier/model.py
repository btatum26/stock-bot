import numpy as np
import pandas as pd
from context import Context
from engine.core.controller import SignalModel


class VolatilityRegimeModel(SignalModel):
    """
    Volatility Regime Model V3
    ==========================
    Long-only swing strategy. Enters trend-following setups in low-vol regimes,
    filtered by EMA slope and support/resistance levels for better risk/reward.

    Regimes (ATR-14 percentile over `atr_lookback`):
        LOW_VOL   <  low_vol_threshold    -> eligible to enter long
        NORMAL    in [low, high]          -> hold if already long, do not enter
        HIGH_VOL  >= high_vol_threshold   -> force flat immediately

    Entry (all must be true):
        regime == LOW_VOL
        EMA_fast is rising: EMA_fast > EMA_fast.shift(ema_slope_period)
        Close > EMA_fast
        EMA_fast > EMA_slow
        ATR_ROC(5) < atr_roc_max            (vol not expanding sharply)
        Close < nearest_resistance * (1 - resistance_buffer)   (room to run)
        R:R >= min_rr_ratio                 (only when both levels are known)
        bars since last exit >= cooldown_bars

    Exit (first condition that fires wins):
        regime == HIGH_VOL                  (hard regime exit, immediate)
        2 consecutive closes below EMA_fast (trend break, 2-bar confirmation)
        ATR trailing stop: close < peak_since_entry - trail_atr_mult * ATR
        Take-profit: close >= nearest_resistance * (1 - tp_buffer)
        Hard stop fallback: (close - entry) / entry <= -stop_loss
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

        low_thr          = float(params.get("low_vol_threshold",  p.low_vol_threshold))
        high_thr         = float(params.get("high_vol_threshold", p.high_vol_threshold))
        atr_lookback     = int(params.get("atr_lookback",         p.atr_lookback))
        atr_roc_max      = float(params.get("atr_roc_max",        p.atr_roc_max))
        stop_loss        = float(params.get("stop_loss",          p.stop_loss))
        cooldown_bars    = int(params.get("cooldown_bars",        3))
        min_rr_ratio     = float(params.get("min_rr_ratio",       1.5))
        resistance_buf   = float(params.get("resistance_buffer",  0.02))
        # sr_stop_buffer retired — ATR trailing stop covers this role
        ema_slope_period = int(params.get("ema_slope_period",     10))
        trail_atr_mult   = float(params.get("trail_atr_mult",     3.0))
        trail_min_gain   = float(params.get("trail_min_gain",    0.05))  # activate only after +5%
        tp_buffer        = float(params.get("tp_buffer",          0.005))
        ema_exit_bars    = int(params.get("ema_exit_bars",        2))

        atr      = df[f.AVERAGETRUERANGE_14]
        ema_fast = df[f.MOVINGAVERAGE_50_EMA]
        ema_slow = df[f.MOVINGAVERAGE_200_EMA]
        close    = df['close'] if 'close' in df.columns else df['Close']

        nearest_support    = df[f.SUPPORTRESISTANCE_0_02_Bill_Williams_1_0_0_015_35_NEAREST_SUPPORT_LEVEL]
        nearest_resistance = df[f.SUPPORTRESISTANCE_0_02_Bill_Williams_1_0_0_015_35_NEAREST_RESISTANCE_LEVEL]

        atr_pct = atr.rolling(atr_lookback, min_periods=atr_lookback).apply(
            lambda w: (w <= w[-1]).mean(), raw=True
        )
        atr_roc_5 = atr / atr.shift(5) - 1.0

        low_vol    = atr_pct < low_thr
        high_vol   = atr_pct >= high_thr
        ema_rising = ema_fast > ema_fast.shift(ema_slope_period)
        uptrend    = (close > ema_fast) & (ema_fast > ema_slow) & ema_rising
        vol_stable = atr_roc_5 < atr_roc_max

        # S/R entry filters — passthrough when levels are unavailable
        has_resistance = nearest_resistance.notna() & (nearest_resistance > close)
        has_support    = nearest_support.notna()    & (nearest_support    < close)

        room_to_run = ~has_resistance | (nearest_resistance > close * (1 + resistance_buf))

        rr_ok    = pd.Series(True, index=df.index)
        both_sr  = has_resistance & has_support
        with np.errstate(divide='ignore', invalid='ignore'):
            rr_val = (nearest_resistance - close) / (close - nearest_support)
        rr_ok[both_sr] = rr_val[both_sr] >= min_rr_ratio

        entry_ok = low_vol & uptrend & vol_stable & room_to_run & rr_ok

        # Arrays for the state machine
        close_arr    = close.to_numpy()
        atr_arr      = atr.to_numpy()
        entry_arr    = entry_ok.fillna(False).to_numpy()
        high_vol_arr = high_vol.fillna(False).to_numpy()
        below_fast   = (close < ema_fast).fillna(False).to_numpy()
        res_arr     = nearest_resistance.to_numpy()
        has_res_arr = has_resistance.to_numpy()

        n   = len(df)
        out = np.zeros(n, dtype=np.float64)

        in_pos         = False
        entry_price    = 0.0
        peak_close     = 0.0
        cooldown       = 0
        bars_below_ema = 0

        for i in range(n):
            if cooldown > 0:
                cooldown -= 1

            if in_pos:
                c = close_arr[i]

                # Update trailing stop peak
                if c > peak_close:
                    peak_close = c

                # --- Exit conditions ---
                # 1. High-vol regime: exit immediately
                force_exit = high_vol_arr[i]

                # 2. EMA trend break: require ema_exit_bars consecutive closes below
                if below_fast[i]:
                    bars_below_ema += 1
                else:
                    bars_below_ema = 0
                ema_break = bars_below_ema >= ema_exit_bars

                # 3. ATR trailing stop — only activates once trade is up trail_min_gain
                atr_val        = atr_arr[i]
                gain_so_far    = (peak_close - entry_price) / entry_price
                trail_active   = gain_so_far >= trail_min_gain
                trail_stop     = trail_active and not np.isnan(atr_val) and (c < peak_close - trail_atr_mult * atr_val)

                # 4. Take-profit: price reached nearest resistance
                take_profit = has_res_arr[i] and (c >= res_arr[i] * (1 - tp_buffer))

                # 5. Hard stop fallback
                hard_stop = (c - entry_price) / entry_price <= -stop_loss

                if force_exit or ema_break or trail_stop or take_profit or hard_stop:
                    in_pos         = False
                    entry_price    = 0.0
                    peak_close     = 0.0
                    bars_below_ema = 0
                    cooldown       = cooldown_bars
                else:
                    out[i] = 1.0
            else:
                bars_below_ema = 0
                if entry_arr[i] and cooldown == 0:
                    in_pos      = True
                    entry_price = close_arr[i]
                    peak_close  = close_arr[i]
                    out[i]      = 1.0

        return pd.Series(out, index=df.index)
