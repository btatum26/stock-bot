import pandas as pd
import numpy as np
from context import Context
from engine.core.controller import SignalModel


def _precompute_rolling(high_arr, low_arr, close_arr, atr_arr, max_days):
    """Precompute rolling max-high, min-low, and mean-ATR for all window sizes."""
    n = len(high_arr)

    # rolling_high[w, i] = max(high[i-w+1 : i+1]) for window size w
    # We only need windows from min_days to max_days, but precompute
    # using a sliding window approach for each needed size.
    # Instead, precompute cumulative structures for O(1) range queries.

    # For range max/min, use a sparse table (O(n log n) build, O(1) query).
    # For mean ATR, use prefix sums.

    # --- Prefix sums for ATR ---
    atr_cumsum = np.zeros(n + 1)
    atr_cumsum[1:] = np.nancumsum(atr_arr)

    # --- Sparse table for range max(high) and min(low) ---
    log2 = max(1, int(np.log2(max_days)) + 1)
    sparse_high = np.full((log2 + 1, n), -np.inf)
    sparse_low = np.full((log2 + 1, n), np.inf)
    sparse_high[0] = high_arr
    sparse_low[0] = low_arr

    k = 1
    while (1 << k) <= max_days and (1 << k) <= n:
        half = 1 << (k - 1)
        for i in range(n - (1 << k) + 1):
            sparse_high[k, i] = max(sparse_high[k-1, i], sparse_high[k-1, i + half])
            sparse_low[k, i] = min(sparse_low[k-1, i], sparse_low[k-1, i + half])
        k += 1

    # --- Daily range for spike detection ---
    daily_range = high_arr - low_arr

    # --- Prefix sums for daily_range max per window (use rolling max) ---
    # Actually for spike detection we need max(daily_range[start:end+1]).
    # Build a sparse table for daily_range too.
    sparse_dr = np.full((log2 + 1, n), -np.inf)
    sparse_dr[0] = daily_range
    k = 1
    while (1 << k) <= max_days and (1 << k) <= n:
        half = 1 << (k - 1)
        for i in range(n - (1 << k) + 1):
            sparse_dr[k, i] = max(sparse_dr[k-1, i], sparse_dr[k-1, i + half])
        k += 1

    # --- Midpoint cross counts: precompute using vectorized approach ---
    # For each bar, compute sign(close - rolling_midpoint). Crosses are
    # detected as sign changes. We'll compute this per-window in the main loop
    # but use numpy diff on the sign array.
    close_sign_above = np.empty(n)  # placeholder, filled per-window

    return atr_cumsum, sparse_high, sparse_low, sparse_dr, daily_range


def _sparse_query_max(sparse, left, right):
    """Range max query on sparse table, inclusive [left, right]."""
    length = right - left + 1
    if length <= 0:
        return -np.inf
    k = int(np.log2(length))
    return max(sparse[k, left], sparse[k, right - (1 << k) + 1])


def _sparse_query_min(sparse, left, right):
    """Range min query on sparse table, inclusive [left, right]."""
    length = right - left + 1
    if length <= 0:
        return np.inf
    k = int(np.log2(length))
    return min(sparse[k, left], sparse[k, right - (1 << k) + 1])


class ConsolidationBreakoutModel(SignalModel):

    def generate_signals(self, df: pd.DataFrame, ctx: Context, params: dict = None, artifacts: dict = None) -> pd.Series:
        p = ctx.params
        close_col = "Close" if "Close" in df.columns else "close"
        high_col = "High" if "High" in df.columns else "high"
        low_col = "Low" if "Low" in df.columns else "low"
        close = df[close_col]
        high = df[high_col]
        low = df[low_col]
        atr = df[ctx.features.AVERAGETRUERANGE_14]
        volume = df[ctx.features.VOLUME]
        spy_regime = df[ctx.features.TICKER_COMPARE_SPY_200_REGIME]

        vol_sma20 = volume.rolling(20).mean()

        min_days = int(p.min_consol_days)
        max_days = int(p.max_consol_days)
        range_thresh = float(p.range_pct_threshold)
        atr_thresh = float(p.atr_contract_thresh)
        bo_margin = float(p.breakout_margin)
        vol_surge = float(p.volume_surge)
        require_vol = bool(int(p.require_volume))
        use_regime = bool(int(p.use_regime_filter))
        base_mult = float(p.base_atr_mult)
        l_scale = float(p.length_scale)
        max_mult = float(p.max_atr_mult)
        min_mult = float(p.min_atr_mult)
        ref_len = int(p.ref_length)
        max_hold = int(p.max_hold_days)
        fb_days = int(p.failed_breakout_days)

        idx = df.index
        n = len(df)

        high_arr = high.values.astype(np.float64)
        low_arr = low.values.astype(np.float64)
        close_arr = close.values.astype(np.float64)
        atr_arr = atr.values.astype(np.float64)
        vol_arr = volume.values.astype(np.float64)
        vol_sma_arr = vol_sma20.values.astype(np.float64)
        regime_arr = spy_regime.values.astype(np.float64)
        sig_arr = np.zeros(n)

        # Precompute data structures for O(1) range queries
        atr_cumsum, sparse_high, sparse_low, sparse_dr, daily_range = \
            _precompute_rolling(high_arr, low_arr, close_arr, atr_arr, max_days + 21)

        # Precompute: sign of (close - midpoint) changes need per-window midpoint,
        # so we can't fully precompute crosses. But we CAN precompute the
        # "above midpoint" sign array once we know the midpoint.
        # The key optimization: use numpy slicing instead of Python inner loops.

        # State tracking for position management
        in_position = False
        entry_bar = 0
        consol_high = 0.0
        consol_low = 0.0
        consol_len = 0
        highest_high = 0.0
        tolerance_mult = 0.0
        atr_at_entry = 0.0

        start_bar = max(max_days + 20, 0)

        for i in range(start_bar, n):
            if in_position:
                if high_arr[i] > highest_high:
                    highest_high = high_arr[i]

                bars_held = i - entry_bar

                if bars_held <= fb_days and close_arr[i] < consol_high:
                    in_position = False
                    continue

                if bars_held > max_hold:
                    in_position = False
                    continue

                exit_level = highest_high - (tolerance_mult * atr_at_entry)
                if exit_level < consol_low:
                    exit_level = consol_low

                if close_arr[i] < exit_level:
                    in_position = False
                    continue

                sig_arr[i] = 1.0
                continue

            # --- Not in position: scan for consolidation + breakout ---

            if use_regime and (np.isnan(regime_arr[i]) or regime_arr[i] < 1.0):
                continue

            # Quick pre-filter: today's close must be above *something* to break out.
            # Skip the expensive scan if close is below the rolling min high.
            end = i - 1
            if end < min_days:
                continue

            best_len = 0
            best_high = 0.0
            best_low = 0.0
            best_atr = 0.0
            det_atr = atr_arr[end]

            if np.isnan(det_atr) or det_atr == 0:
                continue

            # Try windows from longest to shortest — once we find a valid one,
            # we know it's the longest and can stop.
            for w_len in range(min(max_days, end + 1), min_days - 1, -1):
                start = end - w_len + 1
                if start < 0:
                    continue

                # O(1) range queries via sparse tables
                w_high = _sparse_query_max(sparse_high, start, end)
                w_low = _sparse_query_min(sparse_low, start, end)
                w_mid = (w_high + w_low) * 0.5
                if w_mid == 0:
                    continue

                w_range_pct = (w_high - w_low) / w_mid
                if w_range_pct > range_thresh:
                    continue

                # ATR contraction — O(1) via prefix sums
                atr_sum_window = atr_cumsum[end + 1] - atr_cumsum[start]
                w_atr = atr_sum_window / w_len

                prior_start = max(0, start - 20)
                prior_end = start - 1
                if prior_end < prior_start:
                    continue
                prior_len = prior_end - prior_start + 1
                prior_atr_sum = atr_cumsum[prior_end + 1] - atr_cumsum[prior_start]
                prior_atr = prior_atr_sum / prior_len
                if prior_atr == 0 or np.isnan(prior_atr):
                    continue
                if w_atr > prior_atr * atr_thresh:
                    continue

                # Spike check — O(1) via sparse table on daily range
                max_dr = _sparse_query_max(sparse_dr, start, end)
                if max_dr > det_atr * 2.5:
                    continue

                # Bull flag bias: close in top 40% of range
                range_height = w_high - w_low
                if range_height == 0:
                    continue
                close_pos = (close_arr[end] - w_low) / range_height
                if close_pos < 0.40:
                    continue

                # Midpoint crosses — vectorized with numpy
                window_closes = close_arr[start:end + 1]
                above_mid = window_closes > w_mid
                crosses = np.count_nonzero(np.diff(above_mid.astype(np.int8)))
                min_crosses = max(2, w_len // 10)
                if crosses < min_crosses:
                    continue

                # Valid! Since we iterate longest-first, this is the best.
                best_len = w_len
                best_high = w_high
                best_low = w_low
                best_atr = det_atr
                break  # longest valid found

            if best_len == 0:
                continue

            # Breakout check
            breakout_level = best_high + bo_margin * best_atr
            if close_arr[i] <= breakout_level:
                continue

            # Extension filter
            if close_arr[i] > best_high + best_atr * 1.5:
                continue

            # Volume confirmation
            if require_vol:
                if np.isnan(vol_sma_arr[i]) or vol_sma_arr[i] == 0:
                    continue
                if vol_arr[i] / vol_sma_arr[i] < vol_surge:
                    continue

            # --- Entry ---
            in_position = True
            entry_bar = i
            consol_high = best_high
            consol_low = best_low
            consol_len = best_len
            highest_high = high_arr[i]
            atr_at_entry = best_atr
            raw_mult = base_mult + (consol_len - ref_len) * l_scale
            tolerance_mult = max(min_mult, min(max_mult, raw_mult))
            sig_arr[i] = 1.0

        return pd.Series(sig_arr, index=idx)
