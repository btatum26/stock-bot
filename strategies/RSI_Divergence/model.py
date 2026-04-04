import numpy as np
import pandas as pd
from context import Context
from engine.core.controller import SignalModel


class RSIDivergenceModel(SignalModel):
    """
    RSI Fractal Divergence Strategy
    ================================
    Uses the generic Fractals feature for swing-point identification and
    the native RSI feature for momentum — keeping both concerns separate.

    Bullish divergence  — price makes a lower low  but RSI makes a higher low.
    Bearish divergence  — price makes a higher high but RSI makes a lower high.

    RSI at the actual swing-point bar
    ----------------------------------
    The Fractals feature confirms a swing at bar T while the actual extreme
    sits at bar T-n (fractal_n bars earlier).  The RSI value that matters for
    divergence is therefore rsi.shift(fractal_n), read at each confirmation bar.

    NOTE: context.params.fractal_n must match the Fractals feature param.
    """

    def train(self, df: pd.DataFrame, context: Context, params: dict) -> dict:
        return {}   # rule-based — no training artefacts

    def generate_signals(
        self,
        df: pd.DataFrame,
        context: Context,
        params: dict,
        artifacts: dict,
    ) -> pd.Series:

        f = context.features
        n         = int(params.get("fractal_n",         context.params.fractal_n))
        min_div   = float(params.get("min_divergence_rsi",  context.params.min_divergence_rsi))
        max_hold  = int(params.get("max_hold_bars",         context.params.max_hold_bars))
        bull_thr  = float(params.get("rsi_bull_threshold",  context.params.rsi_bull_threshold))
        bear_thr  = float(params.get("rsi_bear_threshold",  context.params.rsi_bear_threshold))

        # ------------------------------------------------------------------
        # Pull columns
        # ------------------------------------------------------------------
        rsi       = df[f.RSI_14]
        frac_high = df[f.IS_FRACTAL_HIGH].astype(bool)
        frac_low  = df[f.IS_FRACTAL_LOW].astype(bool)

        cur_hi_px  = df[f.FRACTAL_HIGH_PRICE]   # sparse — NaN except at confirmation
        prev_hi_px = df[f.PREV_FRACTAL_HIGH]     # previous high, forward-filled

        cur_lo_px  = df[f.FRACTAL_LOW_PRICE]
        prev_lo_px = df[f.PREV_FRACTAL_LOW]

        # RSI at the actual swing-point (T-n), read at the confirmation bar (T)
        rsi_at_bar = rsi.shift(n)
        cur_hi_rsi = rsi_at_bar.where(frac_high)
        cur_lo_rsi = rsi_at_bar.where(frac_low)

        # Previous RSI at fractal — grab confirmed-only values, lag by 1, ffill
        high_idx = frac_high[frac_high].index
        prev_hi_rsi_sparse = cur_hi_rsi.loc[high_idx].shift(1)
        prev_hi_rsi = pd.Series(np.nan, index=df.index)
        prev_hi_rsi.loc[high_idx] = prev_hi_rsi_sparse
        prev_hi_rsi = prev_hi_rsi.ffill()

        low_idx = frac_low[frac_low].index
        prev_lo_rsi_sparse = cur_lo_rsi.loc[low_idx].shift(1)
        prev_lo_rsi = pd.Series(np.nan, index=df.index)
        prev_lo_rsi.loc[low_idx] = prev_lo_rsi_sparse
        prev_lo_rsi = prev_lo_rsi.ffill()

        # ------------------------------------------------------------------
        # Bullish divergence
        #   Fractal low confirmed  +  lower price low  +  higher RSI low
        #   +  RSI gap >= min_divergence_rsi  +  not already overbought
        # ------------------------------------------------------------------
        both_lo_valid = cur_lo_px.notna() & prev_lo_px.notna()
        rsi_bull_gap  = cur_lo_rsi - prev_lo_rsi       # positive = higher RSI low

        bull_mask = (
            frac_low
            & both_lo_valid
            & (cur_lo_px  < prev_lo_px)
            & (rsi_bull_gap >= min_div)
            & (rsi <= bull_thr)
        )

        # ------------------------------------------------------------------
        # Bearish divergence
        #   Fractal high confirmed  +  higher price high  +  lower RSI high
        #   +  RSI gap >= min_divergence_rsi  +  not already oversold
        # ------------------------------------------------------------------
        both_hi_valid = cur_hi_px.notna() & prev_hi_px.notna()
        rsi_bear_gap  = prev_hi_rsi - cur_hi_rsi       # positive = lower RSI high

        bear_mask = (
            frac_high
            & both_hi_valid
            & (cur_hi_px  > prev_hi_px)
            & (rsi_bear_gap >= min_div)
            & (rsi >= bear_thr)
        )

        # ------------------------------------------------------------------
        # Conviction: scale divergence gap to [0, 1] (30-pt soft cap)
        # ------------------------------------------------------------------
        SCALE = 30.0
        bull_conviction = (rsi_bull_gap / SCALE).clip(0.0, 1.0)
        bear_conviction = (rsi_bear_gap / SCALE).clip(0.0, 1.0)

        raw = pd.Series(0.0, index=df.index)
        raw[bull_mask] =  bull_conviction[bull_mask]
        raw[bear_mask] = -bear_conviction[bear_mask]

        # If both fire on the same bar (rare), take the stronger signal
        both = bull_mask & bear_mask
        if both.any():
            raw[both] = np.where(
                bull_conviction[both] >= bear_conviction[both],
                bull_conviction[both],
                -bear_conviction[both],
            )

        # ------------------------------------------------------------------
        # Carry signal forward for max_hold_bars bars
        # ------------------------------------------------------------------
        return (
            raw.replace(0.0, np.nan)
               .ffill(limit=max_hold)
               .fillna(0.0)
        )
