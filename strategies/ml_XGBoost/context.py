# AUTO-GENERATED. Do not edit — updated by the GUI when features change.
from dataclasses import dataclass, field


@dataclass(frozen=True)
class FeaturesContext:
    """Typed mapping from strategy features to DataFrame column names."""
    Norm_MovingAverage_50_EMA: str = 'Norm_MovingAverage_50_EMA'
    Norm_MovingAverage_20_EMA: str = 'Norm_MovingAverage_20_EMA'
    Norm_MovingAverage_200_EMA: str = 'Norm_MovingAverage_200_EMA'
    RSI_14: str = 'RSI_14'
    Norm_AverageTrueRange_14: str = 'Norm_AverageTrueRange_14'


@dataclass(frozen=True)
class ParamsContext:
    """Typed strategy hyperparameters."""
    stop_loss: float = 0.1
    take_profit: float = 1000.0
    n_estimators: int = 300
    max_depth: int = 5
    min_child_weight: int = 20
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    subsample: float = 0.8
    reg_alpha: float = 0.0
    gamma: float = 0.0


@dataclass(frozen=True)
class Context:
    """Master context object."""
    features: FeaturesContext = field(default_factory=FeaturesContext)
    params:   ParamsContext   = field(default_factory=ParamsContext)
