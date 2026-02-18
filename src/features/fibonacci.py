from typing import Dict, Any, List
import pandas as pd
from .base import Feature, FeatureOutput, LevelOutput

class FibonacciRetracement(Feature):
    @property
    def name(self) -> str:
        return "Fibonacci Retracement"

    @property
    def description(self) -> str:
        return "Horizontal levels based on the High-Low range of the visible period."

    @property
    def category(self) -> str:
        return "Price Levels"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "lookback": 0, # 0 means "entire loaded range", otherwise N bars
            "color": "#ffaa00"
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> List[FeatureOutput]:
        lookback = int(params.get("lookback", 0))
        
        if lookback > 0 and len(df) > lookback:
            window = df.iloc[-lookback:]
        else:
            window = df
            
        high_price = window['High'].max()
        low_price = window['Low'].min()
        diff = high_price - low_price
        
        levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        
        outputs = []
        for lvl in levels:
            price = high_price - (diff * lvl)
            outputs.append(LevelOutput(
                name=f"Fib {lvl:.3f}",
                price=price,
                min_price=price * 0.9995, # Tiny visual width
                max_price=price * 1.0005,
                color=params.get("color"),
                strength=1.0
            ))
            
        return outputs
