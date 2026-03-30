
from dataclasses import dataclass, field

@dataclass(frozen=True)
class FeaturesContext:
    SMA_50_close: str = 'SMA_50_close'
    RSI_close_14: str = 'RSI_close_14'
    MACD_12_9_26_close_HIST: str = 'MACD_12_9_26_close_HIST'

@dataclass(frozen=True)
class ParamsContext:
    stop_loss: float = 0.05
    take_profit: float = 0.1

@dataclass(frozen=True)
class Context:
    features: FeaturesContext = field(default_factory=FeaturesContext)
    params: ParamsContext = field(default_factory=ParamsContext)
