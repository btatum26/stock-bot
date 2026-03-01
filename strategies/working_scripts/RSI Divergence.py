import pandas as pd
import numpy as np
from src.signals.base import SignalModel

class StrategySignal(SignalModel):
    def generate_signals(self, df: pd.DataFrame, feature_data: dict) -> pd.Series:
        """
        df: DataFrame with OHLCV data ('Open', 'High', 'Low', 'Close', 'Volume')
        feature_data: Dict of feature results (e.g. {'RSI_14': Series, 'ATR_14': Series})
        """
        signals = pd.Series(0, index=df.index)
        
        # 1. Extract features (Default periods 14)
        rsi = None
        atr = None
        
        # Try to find RSI and ATR in feature_data
        for key in feature_data.keys():
            if key.startswith('RSI'):
                rsi = feature_data[key]
            if key.startswith('ATR'):
                atr = feature_data[key]
        
        if rsi is None or atr is None:
            # If features are missing, we can't generate signals
            # Note: User must add RSI and ATR features in the GUI first
            return signals

        # 2. Strategy Parameters
        oversold_threshold = 30
        atr_multiplier = 3.0
        
        # 3. Iterate to manage state (Trailing Stop)
        in_position = False
        highest_high = 0
        stop_level = 0
        
        # We start from index 2 to allow for i-1 and i-2 checks
        for i in range(2, len(df)):
            if not in_position:
                # Entry Logic: Bullish RSI Reversal
                # 1. Previous RSI was in oversold territory
                # 2. RSI has turned upwards (Current > Previous and Previous <= 2-bars-ago)
                is_oversold = rsi.iloc[i-1] < oversold_threshold
                is_reversal = rsi.iloc[i] > rsi.iloc[i-1] and rsi.iloc[i-1] <= rsi.iloc[i-2]
                
                if is_oversold and is_reversal:
                    signals.iloc[i] = 1 # BUY
                    in_position = True
                    highest_high = df['High'].iloc[i]
                    stop_level = highest_high - (atr_multiplier * atr.iloc[i])
            else:
                # Exit Logic: ATR Trailing Stop (Chandelier Exit)
                # Update highest high since entry
                highest_high = max(highest_high, df['High'].iloc[i])
                
                # Update trailing stop level (it can only move up)
                current_stop = highest_high - (atr_multiplier * atr.iloc[i])
                stop_level = max(stop_level, current_stop)
                
                # Check for exit (Price drops below stop level)
                # We check if Low of current bar crossed the stop level
                if df['Low'].iloc[i] < stop_level:
                    signals.iloc[i] = -1 # SELL
                    in_position = False
                    stop_level = 0
                    highest_high = 0
                    
        return signals
