# AUTO-GENERATED FILE. DO NOT MODIFY DIRECTLY.
from dataclasses import dataclass, field

@dataclass(frozen=True)
class FeaturesContext:
    """Strictly typed mapping of strategy features to DataFrame columns."""
    RSI_2: str = 'RSI_2'
    """
    Feature: RSI
        Outputs: Primary
        Params: {'period': 2}
    """
    RSI_14: str = 'RSI_14'
    """
    Feature: RSI
        Outputs: Primary
        Params: {'period': 14}
    """
    SMA_200: str = 'SMA_200'
    """
    Feature: Simple Moving Average
        Outputs: Primary
        Params: {'period': 200}
    """
    AVERAGETRUERANGE_5: str = 'AverageTrueRange_5'
    """
    Feature: ATR
        Outputs: Primary
        Params: {'period': 5}
    """
    AVERAGETRUERANGE_21: str = 'AverageTrueRange_21'
    """
    Feature: ATR
        Outputs: Primary
        Params: {'period': 21}
    """

@dataclass(frozen=True)
class ParamsContext:
    """Strictly typed strategy hyperparameters defined in manifest.json."""
    rsi_entry: int = 10
    rsi_exit: int = 50
    atr_ratio_max: float = 1.2
    max_hold_days: int = 5
    stop_loss: float = 0.07
    regime_prob_threshold: float = 0.6

@dataclass(frozen=True)
class Context:
    """Master strategy context object."""
    features: FeaturesContext = field(default_factory=FeaturesContext)
    params: ParamsContext = field(default_factory=ParamsContext)