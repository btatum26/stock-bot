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
        n            = int(params.get("fractal_n",            context.params.fractal_n))
        min_div      = float(params.get("min_divergence_rsi", context.params.min_divergence_rsi))
        max_hold     = int(params.get("max_hold_bars",        context.params.max_hold_bars))
        bull_thr     = float(params.get("rsi_bull_threshold", context.params.rsi_bull_threshold))
        bear_thr     = float(params.get("rsi_bear_threshold", context.params.rsi_bear_threshold))
        # NOTE: rsi_bull_threshold defaults to 70 in context — set it to 45-50 in the GUI
        # for meaningful oversold filtering.
        support_tol  = float(params.get("support_tolerance_pct", 0.015))   # ±1.5% of support level
        min_bars     = int(params.get("min_bars_since_high",      5))

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

        # Prefer nearest_support_level (closest active support below price) when available.
        # Falls back to last_support_level if the manifest hasn't been synced yet.
        _near_supp_col = getattr(f, "NEAREST_SUPPORT_LEVEL", None)
        _near_supp_str_col = getattr(f, "NEAREST_SUPPORT_STRENGTH", None)
        if _near_supp_col and _near_supp_col in df.columns:
            support          = df[_near_supp_col]
            support_strength = df[_near_supp_str_col] if _near_supp_str_col and _near_supp_str_col in df.columns else pd.Series(1.0, index=df.index)
        else:
            support          = df[f.LAST_SUPPORT_LEVEL]
            support_strength = pd.Series(1.0, index=df.index)

        bars_since_hi  = df[f.BARS_SINCE_LAST_HIGH]

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
        #   +  RSI gap >= min_divergence_rsi  +  RSI below threshold
        #   +  price near support  +  enough bars since last swing high
        # ------------------------------------------------------------------
        both_lo_valid = cur_lo_px.notna() & prev_lo_px.notna()
        rsi_bull_gap  = cur_lo_rsi - prev_lo_rsi       # positive = higher RSI low

        # Support confluence: fractal low must land within ±support_tol of the last support level.
        support_valid = support.notna() & (support > 0)
        near_support  = (
            support_valid
            & (cur_lo_px >= support * (1.0 - support_tol))
            & (cur_lo_px <= support * (1.0 + support_tol))
        )

        # Trend exhaustion: require a minimum downtrend duration before calling a reversal.
        enough_bars = bars_since_hi.notna() & (bars_since_hi >= min_bars)

        bull_mask = (
            frac_low
            & both_lo_valid
            & (cur_lo_px  < prev_lo_px)
            & (rsi_bull_gap >= min_div)
            & (rsi <= bull_thr)
            & near_support
            & enough_bars
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
        # Bull signals get a proximity boost: exactly at support = 1.0x,
        # at the edge of the tolerance band = 0.5x.
        # ------------------------------------------------------------------
        SCALE = 30.0
        rsi_bull_score = (rsi_bull_gap / SCALE).clip(0.0, 1.0)

        support_dist   = (cur_lo_px - support).abs() / support.replace(0, np.nan)
        proximity      = (1.0 - (support_dist / support_tol).clip(0.0, 1.0)).fillna(0.0)

        # Strength bonus: cap at 5 touches → 1.0, scale 0.8–1.0 so a weak level doesn't zero out
        strength_factor = (support_strength.clip(upper=5.0) / 5.0).clip(0.0, 1.0) * 0.2 + 0.8
        bull_conviction = rsi_bull_score * (0.5 + 0.5 * proximity) * strength_factor

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
