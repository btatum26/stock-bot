from typing import Dict, Any, List
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from sklearn.cluster import AgglomerativeClustering
from .base import Feature, FeatureOutput, LevelOutput

class SupportResistance(Feature):
    @property
    def name(self) -> str:
        return "Support & Resistance"

    @property
    def description(self) -> str:
        return "Identifies key price clusters using Local Extrema, Fractals, or ZigZag."

    @property
    def category(self) -> str:
        return "Price Levels"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "method": ["ZigZag", "Savitzky-Golay", "Bill Williams"], 
            "threshold_pct": 0.015, # For identifying the pivot itself (ZigZag)
            "window": 5, # Only for SG or BW
            "clustering_pct": 0.02, # For merging nearby pivots
            "min_strength": 1.0
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> List[FeatureOutput]:
        method = params.get("method", "ZigZag")
        threshold = float(params.get("threshold_pct", 0.015))
        window = int(params.get("window", 5))
        cluster_thresh = float(params.get("clustering_pct", 0.02))
        min_str = float(params.get("min_strength", 1.0))

        pivots = []
        if method == "Savitzky-Golay":
            pivots = self.get_pivots_smoothed(df, window=window)
        elif method == "Bill Williams":
            pivots = self.get_pivots_bill_williams(df, window=window)
        else: # ZigZag
            pivots = self.get_pivots_zigzag(df, deviation_pct=threshold) 

        clusters = self.cluster_pivots(pivots, cluster_thresh)
        
        outputs = []
        for c in clusters:
            if c['strength'] >= min_str:
                outputs.append(LevelOutput(
                    name=f"Level {c['price']}",
                    price=c['price'],
                    min_price=c['min_price'],
                    max_price=c['max_price'],
                    strength=c['strength'],
                    color='#0000ff' # Blue base
                ))
                
        return outputs

    # --- Analysis Logic ---

    def get_pivots_smoothed(self, df, window=5, polyorder=3):
        # Ensure window is odd and smaller than data
        if window % 2 == 0: window += 1
        if len(df) <= window: return []

        # Smooth Highs and Lows
        smoothed_high = savgol_filter(df['High'], window, polyorder)
        smoothed_low = savgol_filter(df['Low'], window, polyorder)
        
        pivots = []
        half = window // 2
        for i in range(half, len(df) - half):
            # Support
            if smoothed_low[i] == min(smoothed_low[i-half:i+half+1]):
                pivots.append({'price': df['Low'].iloc[i], 'index': i, 'type': 'support'})
            # Resistance
            if smoothed_high[i] == max(smoothed_high[i-half:i+half+1]):
                pivots.append({'price': df['High'].iloc[i], 'index': i, 'type': 'resistance'})
        return pivots

    def get_pivots_bill_williams(self, df, window=2):
        pivots = []
        for i in range(window, len(df) - window):
            # Bullish Fractal (Support)
            is_support = True
            for j in range(1, window + 1):
                if df['Low'].iloc[i] >= df['Low'].iloc[i-j] or df['Low'].iloc[i] >= df['Low'].iloc[i+j]:
                    is_support = False
                    break
            if is_support:
                pivots.append({'price': df['Low'].iloc[i], 'index': i, 'type': 'support'})
            
            # Bearish Fractal (Resistance)
            is_resistance = True
            for j in range(1, window + 1):
                if df['High'].iloc[i] <= df['High'].iloc[i-j] or df['High'].iloc[i] <= df['High'].iloc[i+j]:
                    is_resistance = False
                    break
            if is_resistance:
                pivots.append({'price': df['High'].iloc[i], 'index': i, 'type': 'resistance'})
        return pivots

    def get_pivots_zigzag(self, df, deviation_pct=0.05):
        pivots = []
        last_pivot_price = df['Close'].iloc[0]
        last_pivot_type = None 
        
        for i in range(1, len(df)):
            price_high = df['High'].iloc[i]
            price_low = df['Low'].iloc[i]
            
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
        if not pivots:
            return []

        prices = np.array([p['price'] for p in pivots]).reshape(-1, 1)
        avg_price = np.mean(prices)
        dist_threshold = avg_price * threshold_pct

        model = AgglomerativeClustering(
            n_clusters=None, 
            distance_threshold=dist_threshold, 
            linkage='complete'
        )
        
        clusters = model.fit_predict(prices)
        levels = []
        for cluster_id in np.unique(clusters):
            cluster_prices = prices[clusters == cluster_id]
            level_price = np.mean(cluster_prices)
            
            levels.append({
                'price': round(float(level_price), 2),
                'min_price': round(float(np.min(cluster_prices)), 2),
                'max_price': round(float(np.max(cluster_prices)), 2),
                'strength': len(cluster_prices)
            })
            
        return sorted(levels, key=lambda x: x['strength'], reverse=True)
