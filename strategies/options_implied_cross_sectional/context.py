# AUTO-GENERATED FILE. DO NOT MODIFY DIRECTLY.
from dataclasses import dataclass, field

@dataclass(frozen=True)
class FeaturesContext:
    """Strictly typed mapping of strategy features to DataFrame columns."""
    AVERAGETRUERANGE_14: str = 'AverageTrueRange_14'
    """
    Feature: ATR
        Outputs: Primary
        Params: {'period': 14}
    """
    SMA_50: str = 'SMA_50'
    """
    Feature: Simple Moving Average
        Outputs: Primary
        Params: {'period': 50}
    """

@dataclass(frozen=True)
class ParamsContext:
    """Strictly typed strategy hyperparameters defined in manifest.json."""
    pc_ratio_lookback: int = 5
    iv_skew_lookback: int = 60
    unusual_vol_lookback: int = 20
    rebalance_freq_days: int = 5
    long_only: bool = True

@dataclass(frozen=True)
class Context:
    """Master strategy context object."""
    features: FeaturesContext = field(default_factory=FeaturesContext)
    params: ParamsContext = field(default_factory=ParamsContext)
