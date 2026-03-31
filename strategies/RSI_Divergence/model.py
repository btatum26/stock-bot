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
        # Example Usage:
        # threshold = ctx.params.my_threshold
        # indicator_data = df[ctx.features.MY_INDICATOR]
        
        signals = pd.Series(0.0, index=df.index)
        
        # --- YOUR LOGIC HERE ---
        
        return signals