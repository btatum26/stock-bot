import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

class LevelsAnalyzer:
    def __init__(self, window=5, threshold_pct=0.01):
        """
        window: Number of bars to check on each side for local extrema.
        threshold_pct: Max distance (as % of price) to group points into a single level.
        """
        self.window = window
        self.threshold_pct = threshold_pct

    def find_pivots(self, df):
        """Identifies local highs and lows."""
        pivots = []
        for i in range(self.window, len(df) - self.window):
            # Check for Support (Local Low)
            is_support = True
            for j in range(i - self.window, i + self.window + 1):
                if df['Low'].iloc[i] > df['Low'].iloc[j]:
                    is_support = False
                    break
            if is_support:
                pivots.append({'price': df['Low'].iloc[i], 'type': 'support', 'index': i})

            # Check for Resistance (Local High)
            is_resistance = True
            for j in range(i - self.window, i + self.window + 1):
                if df['High'].iloc[i] < df['High'].iloc[j]:
                    is_resistance = False
                    break
            if is_resistance:
                pivots.append({'price': df['High'].iloc[i], 'type': 'resistance', 'index': i})
        
        return pivots

    def identify_levels(self, df):
        """Groups pivots into clusters and ranks them by strength."""
        pivots = self.find_pivots(df)
        if not pivots:
            return []

        prices = np.array([p['price'] for p in pivots]).reshape(-1, 1)
        
        # Calculate dynamic distance threshold based on average price
        avg_price = np.mean(prices)
        dist_threshold = avg_price * self.threshold_pct

        # Use Agglomerative Clustering to group nearby prices
        # linkage='complete' ensures no two points in a cluster are further than dist_threshold
        cluster_model = AgglomerativeClustering(
            n_clusters=None, 
            distance_threshold=dist_threshold, 
            linkage='complete'
        )
        
        clusters = cluster_model.fit_predict(prices)
        
        levels = []
        for cluster_id in np.unique(clusters):
            cluster_prices = prices[clusters == cluster_id]
            
            # Strength is the number of pivot points that hit this zone
            strength = len(cluster_prices)
            
            # The level price is the average of the pivots in that cluster
            level_price = np.mean(cluster_prices)
            
            # Determine if it's primarily support or resistance
            cluster_pivots = [pivots[i] for i, cid in enumerate(clusters) if cid == cluster_id]
            support_count = sum(1 for p in cluster_pivots if p['type'] == 'support')
            resistance_count = sum(1 for p in cluster_pivots if p['type'] == 'resistance')
            
            levels.append({
                'price': round(float(level_price), 2),
                'min_price': round(float(np.min(cluster_prices)), 2),
                'max_price': round(float(np.max(cluster_prices)), 2),
                'strength': strength,
                'hits': strength
            })
        
        # Sort levels by price for easy reading
        return sorted(levels, key=lambda x: x['price'])

    def get_strong_levels(self, df, min_strength=2):
        """Returns only levels that have been tested at least N times."""
        all_levels = self.identify_levels(df)
        return [l for l in all_levels if l['strength'] >= min_strength]
