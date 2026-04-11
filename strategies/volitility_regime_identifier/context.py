# AUTO-GENERATED. Do not edit — updated by the GUI when features change.
from dataclasses import dataclass, field


@dataclass(frozen=True)
class FeaturesContext:
    """Typed mapping from strategy features to DataFrame column names."""
    MovingAverage_50_EMA: str = 'MovingAverage_50_EMA'
    MovingAverage_200_EMA: str = 'MovingAverage_200_EMA'
    AverageTrueRange_14: str = 'AverageTrueRange_14'


@dataclass(frozen=True)
class ParamsContext:
    """Typed strategy hyperparameters."""
    low_vol_threshold: float = 0.25
    high_vol_threshold: float = 0.8
    atr_lookback: int = 300
    atr_roc_max: float = 0.1
    stop_loss: float = 0.07


@dataclass(frozen=True)
class Context:
    """Master context object."""
    features: FeaturesContext = field(default_factory=FeaturesContext)
    params:   ParamsContext   = field(default_factory=ParamsContext)
