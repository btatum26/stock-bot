from typing import Dict, Any, List
import pandas as pd
from ..base import Feature, FeatureResult, OutputSchema, OutputType, Pane, register_feature

@register_feature("MACD")
class MACD(Feature):
    """
    Computes the Moving Average Convergence Divergence (MACD).
    
    Returns a three-part oscillator comprising the MACD line, the Signal line, 
    and the Histogram difference between the two.
    """
    
    @property
    def name(self) -> str:
        return "MACD"

    @property
    def description(self) -> str:
        return "Moving Average Convergence Divergence. Includes Signal Line and Histogram."

    @property
    def category(self) -> str:
        return "Oscillators (Momentum)"

    @property
    def output_schema(self) -> List[OutputSchema]:
        return [
            OutputSchema(name=None,     output_type=OutputType.LINE,      pane=Pane.NEW),
            OutputSchema(name="signal", output_type=OutputType.LINE,      pane=Pane.NEW),
            OutputSchema(name="hist",   output_type=OutputType.HISTOGRAM, pane=Pane.NEW),
        ]

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "fast_period": 12,
            "slow_period": 26,
            "signal_period": 9,
            "normalize": ["none", "z_score", "pct_distance", "price_ratio"]
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any], cache: Any = None) -> FeatureResult:
        """
        Calculates the MACD components vectorically.

        Requires 'Close' or 'close' in the input DataFrame. If a FeatureCache is 
        provided, it attempts to fetch underlying EMA calculations to save CPU cycles.
        """
        fast = int(params.get("fast_period", 12))
        slow = int(params.get("slow_period", 26))
        signal = int(params.get("signal_period", 9))
        norm_method = params.get("normalize", "none")
        
        if cache:
            ema_fast = cache.get_series("EMA", {"period": fast}, df)
            ema_slow = cache.get_series("EMA", {"period": slow}, df)
        else:
            close = df['Close'] if 'Close' in df.columns else df['close']
            ema_fast = close.ewm(span=fast, adjust=False).mean()
            ema_slow = close.ewm(span=slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        col_macd = self.generate_column_name("MACD", params)
        col_signal = self.generate_column_name("MACD", params, "signal")
        col_hist = self.generate_column_name("MACD", params, "hist")
        
        # Normalize data
        final_macd = self.normalize(df, macd_line, norm_method)
        final_signal = self.normalize(df, signal_line, norm_method)
        final_hist = self.normalize(df, histogram, norm_method)
        
        data_dict = {
            col_macd: final_macd,
            col_signal: final_signal,
            col_hist: final_hist
        }
        
        return FeatureResult(data=data_dict)
