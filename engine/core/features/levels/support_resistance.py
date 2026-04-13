from typing import Dict, Any, List
import bisect
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from sklearn.cluster import AgglomerativeClustering
from ..base import Feature, FeatureResult, OutputSchema, OutputType, Pane, register_feature

@register_feature("SupportResistance")
class SupportResistance(Feature):
    @property
    def name(self) -> str:
        return "Support & Resistance"

    @property
    def description(self) -> str:
        return "Rolling ML-Safe Pivot Tracker with 4 levels and breakout detection."

    @property
    def category(self) -> str:
        return "Price Levels"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "method": ["ZigZag", "Savitzky-Golay", "Bill Williams"],
            "threshold_pct": 0.015,
            "window": 3,
            "clustering_pct": 0.02,
            "min_strength": 1.0
        }

    @property
    def output_schema(self) -> List[OutputSchema]:
        return [
            OutputSchema(name="nearest_support_level",    output_type=OutputType.LINE,   pane=Pane.OVERLAY),
            OutputSchema(name="second_support_level",     output_type=OutputType.LINE,   pane=Pane.OVERLAY),
            OutputSchema(name="nearest_resistance_level", output_type=OutputType.LINE,   pane=Pane.OVERLAY),
            OutputSchema(name="second_resistance_level",  output_type=OutputType.LINE,   pane=Pane.OVERLAY),
            OutputSchema(name="breakout_resistance",      output_type=OutputType.MARKER, pane=Pane.OVERLAY),
            OutputSchema(name="breakdown_support",        output_type=OutputType.MARKER, pane=Pane.OVERLAY),
            OutputSchema(name="levels",                   output_type=OutputType.LEVEL,  pane=Pane.OVERLAY),
        ]

    def compute(self, df: pd.DataFrame, params: Dict[str, Any], cache: Any = None) -> FeatureResult:
        # Normalize column names: GUI capitalizes, engine uses lowercase
        df = df.rename(columns={'High': 'high', 'Low': 'low', 'Close': 'close',
                                 'Open': 'open', 'Volume': 'volume'})

        method         = params.get("method") or "Bill Williams"
        threshold      = float(params.get("threshold_pct") or 0.015)
        window         = int(params.get("window") or 3)
        clustering_pct = float(params.get("clustering_pct") or 0.02)

        # Extract price pivots based on the selected method
        if method == "Savitzky-Golay":
            pivots = self.get_pivots_smoothed(df, window=5)
        elif method == "Bill Williams":
            pivots = self.get_pivots_bill_williams_vectorized(df, window=window)
        else:
            pivots = self.get_pivots_zigzag(df, deviation_pct=threshold)

        # Mark the exact confirmation bar for each pivot (no look-ahead)
        supp_series = pd.Series(np.nan, index=df.index)
        res_series  = pd.Series(np.nan, index=df.index)
        for p in pivots:
            idx = p['index']
            if idx < len(df):
                if p['type'] == 'support':
                    supp_series.iloc[idx] = p['price']
                else:
                    res_series.iloc[idx] = p['price']

        close_vals = df['close'].values
        supp_vals  = supp_series.values
        res_vals   = res_series.values
        n_bars     = len(df)

        near_supp_arr = np.full(n_bars, np.nan)
        near_supp_str = np.zeros(n_bars)
        sec_supp_arr  = np.full(n_bars, np.nan)
        sec_supp_str  = np.zeros(n_bars)
        near_res_arr  = np.full(n_bars, np.nan)
        near_res_str  = np.zeros(n_bars)
        sec_res_arr   = np.full(n_bars, np.nan)
        sec_res_str   = np.zeros(n_bars)

        running_supps: list = []   # maintained in sorted order (ascending)
        running_ress: list  = []   # maintained in sorted order (ascending)

        for i in range(n_bars):
            s = supp_vals[i]
            r = res_vals[i]
            if not np.isnan(s):
                bisect.insort(running_supps, float(s))
            if not np.isnan(r):
                bisect.insort(running_ress, float(r))

            c = float(close_vals[i])

            # --- Nearest support at or below close ---
            pos = bisect.bisect_right(running_supps, c) - 1
            if pos >= 0:
                ns = running_supps[pos]
                near_supp_arr[i] = ns
                lo = bisect.bisect_left(running_supps,  ns * (1.0 - clustering_pct))
                hi = bisect.bisect_right(running_supps, ns * (1.0 + clustering_pct))
                near_supp_str[i] = float(hi - lo)

                # 2nd nearest support: skip all levels that cluster with nearest
                j = pos - 1
                while j >= 0 and running_supps[j] >= ns * (1.0 - clustering_pct):
                    j -= 1
                if j >= 0:
                    ss = running_supps[j]
                    sec_supp_arr[i] = ss
                    lo = bisect.bisect_left(running_supps,  ss * (1.0 - clustering_pct))
                    hi = bisect.bisect_right(running_supps, ss * (1.0 + clustering_pct))
                    sec_supp_str[i] = float(hi - lo)

            # --- Nearest resistance at or above close ---
            pos = bisect.bisect_left(running_ress, c)
            if pos < len(running_ress):
                nr = running_ress[pos]
                near_res_arr[i] = nr
                lo = bisect.bisect_left(running_ress,  nr * (1.0 - clustering_pct))
                hi = bisect.bisect_right(running_ress, nr * (1.0 + clustering_pct))
                near_res_str[i] = float(hi - lo)

                # 2nd nearest resistance: skip all levels that cluster with nearest
                j = pos + 1
                while j < len(running_ress) and running_ress[j] <= nr * (1.0 + clustering_pct):
                    j += 1
                if j < len(running_ress):
                    sr = running_ress[j]
                    sec_res_arr[i] = sr
                    lo = bisect.bisect_left(running_ress,  sr * (1.0 - clustering_pct))
                    hi = bisect.bisect_right(running_ress, sr * (1.0 + clustering_pct))
                    sec_res_str[i] = float(hi - lo)

        nearest_support_level    = pd.Series(near_supp_arr, index=df.index)
        second_support_level     = pd.Series(sec_supp_arr,  index=df.index)
        nearest_resistance_level = pd.Series(near_res_arr,  index=df.index)
        second_resistance_level  = pd.Series(sec_res_arr,   index=df.index)

        # --- Breakout / breakdown detection (no look-ahead) ---
        # Fire when current close crosses through the *previous* bar's nearest level.
        close_s        = df['close']
        prev_near_res  = nearest_resistance_level.shift(1)
        prev_near_supp = nearest_support_level.shift(1)

        breakout_res_mask = (
            prev_near_res.notna() &
            (close_s.shift(1) <= prev_near_res) &
            (close_s > prev_near_res)
        )
        breakout_resistance = pd.Series(
            np.where(breakout_res_mask, 1.0, np.nan), index=df.index
        )

        breakdown_supp_mask = (
            prev_near_supp.notna() &
            (close_s.shift(1) >= prev_near_supp) &
            (close_s < prev_near_supp)
        )
        breakdown_support = pd.Series(
            np.where(breakdown_supp_mask, -1.0, np.nan), index=df.index
        )

        G = self.generate_column_name
        data_dict = {
            G("SupportResistance", params, "nearest_support_level"):    nearest_support_level,
            G("SupportResistance", params, "nearest_support_strength"): pd.Series(near_supp_str, index=df.index),
            G("SupportResistance", params, "second_support_level"):     second_support_level,
            G("SupportResistance", params, "second_support_strength"):  pd.Series(sec_supp_str,  index=df.index),
            G("SupportResistance", params, "nearest_resistance_level"): nearest_resistance_level,
            G("SupportResistance", params, "nearest_resistance_strength"): pd.Series(near_res_str, index=df.index),
            G("SupportResistance", params, "second_resistance_level"):  second_resistance_level,
            G("SupportResistance", params, "second_resistance_strength"): pd.Series(sec_res_str,  index=df.index),
            G("SupportResistance", params, "breakout_resistance"):      breakout_resistance,
            G("SupportResistance", params, "breakdown_support"):        breakdown_support,
        }

        clustered_levels = self.cluster_pivots(pivots, clustering_pct)
        return FeatureResult(data=data_dict, levels=clustered_levels)

    # --- Pivot Extraction Methods ---

    def get_pivots_bill_williams_vectorized(self, df, window=2):
        """Identifies Fractals (pivots) where a point is the highest/lowest in its local window."""
        lows  = df['low']
        highs = df['high']

        is_support = True
        for j in range(1, window + 1):
            is_support &= (lows < lows.shift(j)) & (lows < lows.shift(-j))
        confirmed_support = is_support.shift(window)

        is_resistance = True
        for j in range(1, window + 1):
            is_resistance &= (highs > highs.shift(j)) & (highs > highs.shift(-j))
        confirmed_resistance = is_resistance.shift(window)

        pivots = []
        supp_indices = np.where(confirmed_support == True)[0]
        for idx in supp_indices:
            pivots.append({'price': df['low'].iloc[idx - window], 'index': idx, 'type': 'support'})

        res_indices = np.where(confirmed_resistance == True)[0]
        for idx in res_indices:
            pivots.append({'price': df['high'].iloc[idx - window], 'index': idx, 'type': 'resistance'})

        return sorted(pivots, key=lambda x: x['index'])

    def get_pivots_smoothed(self, df, window=5, polyorder=3):
        """Uses Savitzky-Golay filter to smooth price before identifying local extrema."""
        if window % 2 == 0: window += 1
        if len(df) <= window: return []

        smoothed_high = savgol_filter(df['high'], window, polyorder)
        smoothed_low  = savgol_filter(df['low'],  window, polyorder)

        pivots = []
        half = window // 2
        for i in range(half, len(df) - half):
            confirmation_idx = i + half
            if smoothed_low[i]  == min(smoothed_low[i-half:i+half+1]):
                pivots.append({'price': df['low'].iloc[i],  'index': confirmation_idx, 'type': 'support'})
            if smoothed_high[i] == max(smoothed_high[i-half:i+half+1]):
                pivots.append({'price': df['high'].iloc[i], 'index': confirmation_idx, 'type': 'resistance'})
        return sorted(pivots, key=lambda x: x['index'])

    def get_pivots_zigzag(self, df, deviation_pct=0.05):
        """Standard ZigZag algorithm tracking price swings exceeding a percentage threshold."""
        pivots = []
        last_pivot_price = df['close'].iloc[0]
        last_pivot_type  = None

        for i in range(1, len(df)):
            price_high = df['high'].iloc[i]
            price_low  = df['low'].iloc[i]

            diff_high = (price_high - last_pivot_price) / last_pivot_price
            diff_low  = (price_low  - last_pivot_price) / last_pivot_price

            if last_pivot_type is None:
                if diff_high >= deviation_pct:
                    last_pivot_type  = 'H'
                    last_pivot_price = price_high
                    pivots.append({'price': price_high, 'index': i, 'type': 'resistance'})
                elif diff_low <= -deviation_pct:
                    last_pivot_type  = 'L'
                    last_pivot_price = price_low
                    pivots.append({'price': price_low, 'index': i, 'type': 'support'})
            elif last_pivot_type == 'H':
                if price_high > last_pivot_price:
                    last_pivot_price = price_high
                    pivots[-1] = {'price': price_high, 'index': i, 'type': 'resistance'}
                elif diff_low <= -deviation_pct:
                    last_pivot_type  = 'L'
                    last_pivot_price = price_low
                    pivots.append({'price': price_low, 'index': i, 'type': 'support'})
            elif last_pivot_type == 'L':
                if price_low < last_pivot_price:
                    last_pivot_price = price_low
                    pivots[-1] = {'price': price_low, 'index': i, 'type': 'support'}
                elif diff_high >= deviation_pct:
                    last_pivot_type  = 'H'
                    last_pivot_price = price_high
                    pivots.append({'price': price_high, 'index': i, 'type': 'resistance'})
        return pivots

    def cluster_pivots(self, pivots, threshold_pct):
        """Agglomerative clustering to group nearby pivots into significant support/resistance levels."""
        if not pivots: return []
        if len(pivots) == 1:
            p = pivots[0]
            return [{'value': round(float(p['price']), 2), 'label': p['type'],
                     'min_price': round(float(p['price']), 2), 'max_price': round(float(p['price']), 2),
                     'strength': 1}]

        prices    = np.array([p['price'] for p in pivots]).reshape(-1, 1)
        avg_price = np.mean(prices)
        dist_threshold = avg_price * threshold_pct

        model    = AgglomerativeClustering(n_clusters=None, distance_threshold=dist_threshold, linkage='complete')
        clusters = model.fit_predict(prices)

        levels = []
        for cluster_id in np.unique(clusters):
            cluster_prices = prices[clusters == cluster_id]
            level_price = np.mean(cluster_prices)
            levels.append({
                'value':     round(float(level_price), 2),
                'label':     f"Level {round(float(level_price), 2)}",
                'min_price': round(float(np.min(cluster_prices)), 2),
                'max_price': round(float(np.max(cluster_prices)), 2),
                'strength':  len(cluster_prices)
            })

        return sorted(levels, key=lambda x: x['strength'], reverse=True)
