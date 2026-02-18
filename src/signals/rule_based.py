import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from typing import Dict, List, Any, Optional
from .base import SignalModel

class DivergenceSignalModel(SignalModel):
    """
    Detects divergence between price and an indicator (e.g., RSI).
    """
    def __init__(self, name: str, indicator: str = "RSI_14", lookback: int = 20, order: int = 5):
        super().__init__(name)
        self.indicator = indicator
        self.lookback = lookback
        self.order = order # window size for local extrema

    def _find_extrema(self, series: pd.Series, mode: str = 'min') -> List[int]:
        """
        Finds local minima/maxima in a series.
        """
        if mode == 'min':
            extrema = argrelextrema(series.values, np.less, order=self.order)[0]
        else:
            extrema = argrelextrema(series.values, np.greater, order=self.order)[0]
        return list(extrema)

    def generate_signals(self, df: pd.DataFrame, feature_data: Dict[str, pd.Series]) -> pd.Series:
        if self.indicator not in feature_data:
            return pd.Series(0, index=df.index)
        
        indicator_data = feature_data[self.indicator]
        price_data = df['Close']
        
        signals = pd.Series(0, index=df.index)
        
        # Bullish Divergence: Lower Low in price, Higher Low in indicator
        price_mins = self._find_extrema(price_data, 'min')
        indicator_mins = self._find_extrema(indicator_data, 'min')
        
        # We need at least two local minima
        if len(price_mins) < 2 or len(indicator_mins) < 2:
            return signals

        # For each local minimum in indicator, check if price also has a local minimum nearby
        for i in range(1, len(indicator_mins)):
            curr_idx = indicator_mins[i]
            prev_idx = indicator_mins[i-1]
            
            if (curr_idx - prev_idx) > self.lookback:
                continue

            # Bullish Divergence Check
            if indicator_data.iloc[curr_idx] > indicator_data.iloc[prev_idx]:
                # Now check price at these same (or very close) points
                # For simplicity, we'll check price at the exact same indices
                if price_data.iloc[curr_idx] < price_data.iloc[prev_idx]:
                    signals.iloc[curr_idx] = 1 # BUY Signal

            # Bearish Divergence: Higher High in price, Lower High in indicator
            # Similar logic for maxima... (omitted for brevity but recommended)

        return signals

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "indicator": self.indicator,
            "lookback": self.lookback,
            "order": self.order
        })
        return d
