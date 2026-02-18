import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class SignalEvent:
    name: str
    index: int
    timestamp: Any
    value: float
    side: str # 'buy' or 'sell'
    description: str

class SignalEngine:
    """
    Analyzes feature data to detect specific trading events.
    """
    
    @staticmethod
    def detect_crossover(series1: pd.Series, series2: pd.Series) -> pd.Series:
        """
        Detects where series1 crosses over series2.
        Returns: 1 for cross up, -1 for cross down, 0 otherwise.
        """
        # Ensure indices match
        combined = pd.DataFrame({'s1': series1, 's2': series2}).ffill()
        
        prev_s1 = combined['s1'].shift(1)
        prev_s2 = combined['s2'].shift(1)
        
        cross_up = (prev_s1 <= prev_s2) & (combined['s1'] > combined['s2'])
        cross_down = (prev_s1 >= prev_s2) & (combined['s1'] < combined['s2'])
        
        signals = pd.Series(0, index=combined.index)
        signals[cross_up] = 1
        signals[cross_down] = -1
        return signals

    @staticmethod
    def detect_threshold(series: pd.Series, threshold: float) -> pd.Series:
        """
        Detects where series crosses a fixed threshold.
        """
        prev_val = series.shift(1)
        cross_up = (prev_val <= threshold) & (series > threshold)
        cross_down = (prev_val >= threshold) & (series < threshold)
        
        signals = pd.Series(0, index=series.index)
        signals[cross_up] = 1
        signals[cross_down] = -1
        return signals

    @staticmethod
    def detect_price_above_level(price: pd.Series, level: float) -> pd.Series:
        """
        Returns 1 while price is above level, 0 otherwise.
        """
        return (price > level).astype(int)

    def extract_signals(self, df: pd.DataFrame, feature_data: Dict[str, pd.Series]) -> List[SignalEvent]:
        """
        Example extraction logic for the 5 events.
        """
        events = []
        
        # 1. & 2. Bullish/Bearish MA Crossover (if we have multiple MAs)
        ma_keys = [k for k in feature_data.keys() if "MA" in k]
        if len(ma_keys) >= 2:
            s1, s2 = feature_data[ma_keys[0]], feature_data[ma_keys[1]]
            cross = self.detect_crossover(s1, s2)
            for idx in cross[cross != 0].index:
                iloc = df.index.get_loc(idx)
                side = 'buy' if cross[idx] == 1 else 'sell'
                events.append(SignalEvent(
                    name="MA Crossover",
                    index=iloc,
                    timestamp=idx,
                    value=df['Close'].iloc[iloc],
                    side=side,
                    description=f"{ma_keys[0]} crossed {'above' if side=='buy' else 'below'} {ma_keys[1]}"
                ))

        # 3. & 4. RSI Overbought/Oversold Cross
        rsi_keys = [k for k in feature_data.keys() if "RSI" in k]
        if rsi_keys:
            rsi = feature_data[rsi_keys[0]]
            # Cross below 30 (Oversold exit/entry)
            cross_30 = self.detect_threshold(rsi, 30)
            for idx in cross_30[cross_30 != 0].index:
                iloc = df.index.get_loc(idx)
                if cross_30[idx] == 1: # Crossing UP through 30
                     events.append(SignalEvent("RSI Oversold Exit", iloc, idx, df['Close'].iloc[iloc], 'buy', "RSI crossed above 30"))
            
            # Cross above 70 (Overbought exit/entry)
            cross_70 = self.detect_threshold(rsi, 70)
            for idx in cross_70[cross_70 != 0].index:
                iloc = df.index.get_loc(idx)
                if cross_70[idx] == -1: # Crossing DOWN through 70
                     events.append(SignalEvent("RSI Overbought Exit", iloc, idx, df['Close'].iloc[iloc], 'sell', "RSI crossed below 70"))

        # 5. Price Crossover (Price vs MA)
        if ma_keys:
            ma = feature_data[ma_keys[0]]
            cross_price = self.detect_crossover(df['Close'], ma)
            for idx in cross_price[cross_price != 0].index:
                iloc = df.index.get_loc(idx)
                side = 'buy' if cross_price[idx] == 1 else 'sell'
                events.append(SignalEvent(
                    name="Price/MA Cross",
                    index=iloc,
                    timestamp=idx,
                    value=df['Close'].iloc[iloc],
                    side=side,
                    description=f"Price crossed {'above' if side=='buy' else 'below'} {ma_keys[0]}"
                ))

        return events
