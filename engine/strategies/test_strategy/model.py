
import numpy as np
import pandas as pd
from engine.core.controller import SignalModel  # noqa: F401 (imported for dynamic strategy loading)

class TestStrategy(SignalModel):
    def train(self, df, context, params):
        return {}

    def generate_signals(self, df, context, artifacts=None):
        rsi_val = df[context.features.RSI_close_14]
        sma_val = df[context.features.SMA_50_close]
        macd_hist = df[context.features.MACD_12_9_26_close_HIST]
        
        # Standardize on 'close' (handling potential capitalization differences)
        close_price = df['close'] if 'close' in df.columns else df['Close']
        
        condition_long = (
            (close_price > sma_val) & 
            (rsi_val < 70) & 
            (macd_hist > 0.0)
        )
        
        condition_short = (
            (close_price < sma_val) & 
            (rsi_val > 30) & 
            (macd_hist < 0.0)
        )
        
        signals = np.select(
            [condition_long, condition_short], 
            [1.0, -1.0], 
            default=0.0
        )
        return pd.Series(signals, index=df.index)
