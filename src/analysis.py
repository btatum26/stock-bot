import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.cluster import AgglomerativeClustering

class LevelsAnalyzer:
    def __init__(self, threshold_pct=0.015):
        self.threshold_pct = threshold_pct

    def get_pivots_smoothed(self, df, window=5, polyorder=3):
        """
        Uses Savitzky-Golay filter to smooth the price action before finding pivots.
        """
        # Ensure window is odd and smaller than data
        if window % 2 == 0: window += 1
        if len(df) <= window: return []

        # Smooth Highs and Lows
        smoothed_high = savgol_filter(df['High'], window, polyorder)
        smoothed_low = savgol_filter(df['Low'], window, polyorder)
        
        pivots = []
        # Find local extrema on smoothed data
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
        """
        Bill Williams Fractals: A high/low that is preceded and followed by 
        N (window) lower highs/higher lows. Standard Bill Williams uses window=2 (5-bar pattern).
        """
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
        """
        ZigZag algorithm: identifies significant trend reversals based on 
        a minimum percentage price move.
        """
        pivots = []
        last_pivot_price = df['Close'].iloc[0]
        last_pivot_type = None # 'H' for high, 'L' for low
        
        for i in range(1, len(df)):
            price_high = df['High'].iloc[i]
            price_low = df['Low'].iloc[i]
            
            # Calculate % changes from last pivot
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
                    # Update the peak
                    last_pivot_price = price_high
                    pivots[-1] = {'price': price_high, 'index': i, 'type': 'resistance'}
                elif diff_low <= -deviation_pct:
                    # New trough
                    last_pivot_type = 'L'
                    last_pivot_price = price_low
                    pivots.append({'price': price_low, 'index': i, 'type': 'support'})
            
            elif last_pivot_type == 'L':
                if price_low < last_pivot_price:
                    # Update the trough
                    last_pivot_price = price_low
                    pivots[-1] = {'price': price_low, 'index': i, 'type': 'support'}
                elif diff_high >= deviation_pct:
                    # New peak
                    last_pivot_type = 'H'
                    last_pivot_price = price_high
                    pivots.append({'price': price_high, 'index': i, 'type': 'resistance'})
                    
        return pivots

    def cluster_pivots(self, pivots):
        """
        Groups the found pivots into clusters to identify significant interest zones.
        """
        if not pivots:
            return []

        prices = np.array([p['price'] for p in pivots]).reshape(-1, 1)
        avg_price = np.mean(prices)
        dist_threshold = avg_price * self.threshold_pct

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
