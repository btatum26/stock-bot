# AUTO-GENERATED. Do not edit — updated by the GUI when features change.
from dataclasses import dataclass, field


@dataclass(frozen=True)
class FeaturesContext:
    """Typed mapping from strategy features to DataFrame column names."""
    MovingAverage_50_EMA: str = 'MovingAverage_50_EMA'
    MovingAverage_200_EMA: str = 'MovingAverage_200_EMA'
    AverageTrueRange_14: str = 'AverageTrueRange_14'
    SIN: str = 'YearlyCycle_SIN'
    COS: str = 'YearlyCycle_COS'
    LAST_SUPPORT_LEVEL: str = 'SupportResistance_0.02_Bill Williams_1.0_0.015_35_LAST_SUPPORT_LEVEL'
    LAST_RESISTANCE_LEVEL: str = 'SupportResistance_0.02_Bill Williams_1.0_0.015_35_LAST_RESISTANCE_LEVEL'
    NEAREST_SUPPORT_LEVEL: str = 'SupportResistance_0.02_Bill Williams_1.0_0.015_35_NEAREST_SUPPORT_LEVEL'
    NEAREST_RESISTANCE_LEVEL: str = 'SupportResistance_0.02_Bill Williams_1.0_0.015_35_NEAREST_RESISTANCE_LEVEL'


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
