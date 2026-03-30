import pandas as pd
from context import Context
# TODO: Import your specific Engine Base Class here once defined (e.g., RuleBasedStrategy)

class SignalModel:
    
    def generate_signals(self, df: pd.DataFrame, ctx: Context, artifacts: dict = None) -> pd.Series:
        """
        Executes vectorized signal generation logic.
        
        WARNING: `df` is a read-only view. Do not attempt to assign new columns directly to it.
        Use `ctx.features` to access data safely. Use `ctx.params` to access hyperparameters.
        
        Returns:
            pd.Series: A series of conviction weights bounded between [-1.0, 1.0].
        """
        # 1. Safely access data and parameters
        # Note: Assuming your raw data uses 'Close' with a capital C
        close = df['Close'] 
        rsi = df[ctx.features.RSI_close_14]
        
        oversold = ctx.params.oversold
        overbought = ctx.params.overbought
        
        # Pandas requires integers for rolling windows
        lookback = int(ctx.params.divergence_lookback) 

        # 2. Bullish Divergence Logic
        # Price makes a new low compared to the previous 'lookback' window
        lower_low_price = close < close.shift(1).rolling(lookback).min()
        
        # RSI is higher than it was 'lookback' periods ago, AND it is currently oversold
        higher_low_rsi = (rsi > rsi.shift(lookback)) & (rsi < oversold)
        
        bullish_div = lower_low_price & higher_low_rsi

        # 3. Bearish Divergence Logic
        # Price makes a new high
        higher_high_price = close > close.shift(1).rolling(lookback).max()
        
        # RSI is lower than it was, AND it is currently overbought
        lower_high_rsi = (rsi < rsi.shift(lookback)) & (rsi > overbought)
        
        bearish_div = higher_high_price & lower_high_rsi

        # 4. Generate the Phase 3 Bounded Signal Array
        signals = pd.Series(0.0, index=df.index)
        signals[bullish_div] = 1.0   # Full conviction LONG
        signals[bearish_div] = -1.0  # Full conviction SHORT
        
        return signals