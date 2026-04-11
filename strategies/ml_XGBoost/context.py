# AUTO-GENERATED. Do not edit — updated by the GUI when features change.
from dataclasses import dataclass, field


@dataclass(frozen=True)
class FeaturesContext:
    """Typed mapping from strategy features to DataFrame column names."""
    Norm_MovingAverage_50_EMA: str = 'Norm_MovingAverage_50_EMA'
    Norm_MovingAverage_20_EMA: str = 'Norm_MovingAverage_20_EMA'
    Norm_MovingAverage_200_EMA: str = 'Norm_MovingAverage_200_EMA'
    RSI_54: str = 'RSI_54'
    Norm_AverageTrueRange_21: str = 'Norm_AverageTrueRange_21'
    SIN: str = 'YearlyCycle_SIN'
    COS: str = 'YearlyCycle_COS'


@dataclass(frozen=True)
class ParamsContext:
    """Typed strategy hyperparameters."""
    stop_loss: float = 0.1
    take_profit: float = 1000.0
    n_estimators: int = 300
    max_depth: int = 3
    min_child_weight: int = 70
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.75
    subsample: float = 0.65
    reg_alpha: float = 0.1
    gamma: float = 0.02
    lookforward: int = 15
    up_threshold: float = 0.1
    down_threshold: float = -0.01
    entry_threshold: float = 0.46
    exit_threshold: float = 0.3


@dataclass(frozen=True)
class Context:
    """Master context object."""
    features: FeaturesContext = field(default_factory=FeaturesContext)
    params:   ParamsContext   = field(default_factory=ParamsContext)
