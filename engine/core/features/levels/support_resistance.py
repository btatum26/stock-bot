from typing import Dict, Any, List
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
        return "Rolling ML-Safe Pivot Tracker."

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
            OutputSchema(name="last_support_level",    output_type=OutputType.LINE, pane=Pane.OVERLAY),
            OutputSchema(name="last_resistance_level", output_type=OutputType.LINE, pane=Pane.OVERLAY),
            OutputSchema(name="levels", output_type=OutputType.LEVEL, pane=Pane.OVERLAY),
        ]

    def compute(self, df: pd.DataFrame, params: Dict[str, Any], cache: Any = None) -> FeatureResult:
        # Normalize column names: GUI capitalizes, engine uses lowercase
        df = df.rename(columns={'High': 'high', 'Low': 'low', 'Close': 'close',
                                 'Open': 'open', 'Volume': 'volume'})

        method = params.get("method") or "Bill Williams"
        threshold = float(params.get("threshold_pct") or 0.015)
        window = int(params.get("window") or 3)
        clustering_pct = float(params.get("clustering_pct") or 0.02)

        # Extract price pivots based on the selected method
        pivots = []
        if method == "Savitzky-Golay":
            pivots = self.get_pivots_smoothed(df, window=5)
        elif method == "Bill Williams":
            pivots = self.get_pivots_bill_williams_vectorized(df, window=window)
        else:
            pivots = self.get_pivots_zigzag(df, deviation_pct=threshold)

        # Alpha Engine Data: Generate ML-safe level features
        supp_series = pd.Series(np.nan, index=df.index)
        res_series = pd.Series(np.nan, index=df.index)

        # Mark the exact confirmation bar for each pivot to avoid look-ahead bias
        for p in pivots:
            idx = p['index']
            if idx < len(df):
                if p['type'] == 'support':
                    supp_series.iloc[idx] = p['price']
                else:
                    res_series.iloc[idx] = p['price']

        # Forward-fill confirmed levels to provide a continuous state for strategies
        rolling_supp = supp_series.ffill()
        rolling_res = res_series.ffill()

        # Calculate percentage distance from current price to the last confirmed levels
        dist_to_supp = (df['close'] - rolling_supp) / df['close']
        dist_to_res = (rolling_res - df['close']) / df['close']

        # Standardize column names
        col_dist_supp = self.generate_column_name("SupportResistance", params, "dist_to_support")
        col_dist_res = self.generate_column_name("SupportResistance", params, "dist_to_resistance")
        col_last_supp = self.generate_column_name("SupportResistance", params, "last_support_level")
        col_last_res = self.generate_column_name("SupportResistance", params, "last_resistance_level")

        data_dict = {
            col_dist_supp: dist_to_supp.fillna(0.0),
            col_dist_res: dist_to_res.fillna(0.0),
            col_last_supp: rolling_supp.fillna(df['close']),
            col_last_res: rolling_res.fillna(df['close'])
        }

        # Cluster pivots into significant levels for visualization
        clustered_levels = self.cluster_pivots(pivots, clustering_pct)

        return FeatureResult(data=data_dict, levels=clustered_levels)

    # --- Pivot Extraction Methods ---

    def get_pivots_bill_williams_vectorized(self, df, window=2):
        """Identifies Fractals (pivots) where a point is the highest/lowest in its local window."""
        lows = df['low']
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
        smoothed_low = savgol_filter(df['low'], window, polyorder)

        pivots = []
        half = window // 2
        for i in range(half, len(df) - half):
            confirmation_idx = i + half
            if smoothed_low[i] == min(smoothed_low[i-half:i+half+1]):
                pivots.append({'price': df['low'].iloc[i], 'index': confirmation_idx, 'type': 'support'})
            if smoothed_high[i] == max(smoothed_high[i-half:i+half+1]):
                pivots.append({'price': df['high'].iloc[i], 'index': confirmation_idx, 'type': 'resistance'})
        return sorted(pivots, key=lambda x: x['index'])

    def get_pivots_zigzag(self, df, deviation_pct=0.05):
        """Standard ZigZag algorithm tracking price swings exceeding a percentage threshold."""
        pivots = []
        last_pivot_price = df['close'].iloc[0]
        last_pivot_type = None

        for i in range(1, len(df)):
            price_high = df['high'].iloc[i]
            price_low = df['low'].iloc[i]

            diff_high = (price_high - last_pivot_price) / last_pivot_price
            diff_low = (price_low - last_pivot_price) / last_pivot_price

            if last_pivot_type is None:
                if diff_high >= deviation_pct:
                    last_pivot_type = 'H'
                    last_pivot_price = price_high
                    pivots.append({'price': price_high, 'index': i, 'type': 'resistance'})
                elif diff_low <= -deviation_pct:
                    last_pivot_type = 'L'
                    last_pivot_price = price_low
                    pivots.append({'price': price_low, 'index': i, 'type': 'support'})
            elif last_pivot_type == 'H':
                if price_high > last_pivot_price:
                    last_pivot_price = price_high
                    pivots[-1] = {'price': price_high, 'index': i, 'type': 'resistance'}
                elif diff_low <= -deviation_pct:
                    last_pivot_type = 'L'
                    last_pivot_price = price_low
                    pivots.append({'price': price_low, 'index': i, 'type': 'support'})
            elif last_pivot_type == 'L':
                if price_low < last_pivot_price:
                    last_pivot_price = price_low
                    pivots[-1] = {'price': price_low, 'index': i, 'type': 'support'}
                elif diff_high >= deviation_pct:
                    last_pivot_type = 'H'
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

        prices = np.array([p['price'] for p in pivots]).reshape(-1, 1)
        avg_price = np.mean(prices)
        dist_threshold = avg_price * threshold_pct

        model = AgglomerativeClustering(n_clusters=None, distance_threshold=dist_threshold, linkage='complete')
        clusters = model.fit_predict(prices)

        levels = []
        for cluster_id in np.unique(clusters):
            cluster_prices = prices[clusters == cluster_id]
            level_price = np.mean(cluster_prices)
            levels.append({
                'value': round(float(level_price), 2),
                'label': f"Level {round(float(level_price), 2)}",
                'min_price': round(float(np.min(cluster_prices)), 2),
                'max_price': round(float(np.max(cluster_prices)), 2),
                'strength': len(cluster_prices)
            })

        return sorted(levels, key=lambda x: x['strength'], reverse=True)
