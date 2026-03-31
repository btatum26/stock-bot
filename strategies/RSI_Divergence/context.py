# AUTO-GENERATED FILE. DO NOT MODIFY DIRECTLY.
from dataclasses import dataclass, field

@dataclass(frozen=True)
class FeaturesContext:
    """Strictly typed mapping of strategy features to DataFrame columns."""
    RSI_14: str = 'RSI_14'
    """
    Feature: RSI
        Outputs: Primary
        Params: {'period': 14}
    """
    BOLLINGERBANDS_20_2.0_UPPER: str = 'BollingerBands_20_2.0_UPPER'
    """
    Feature: Bollinger Bands
        Outputs: upper
        Params: {'period': 20, 'std_dev': 2.0}
    """
    BOLLINGERBANDS_20_2.0_MID: str = 'BollingerBands_20_2.0_MID'
    """
    Feature: Bollinger Bands
        Outputs: mid
        Params: {'period': 20, 'std_dev': 2.0}
    """
    BOLLINGERBANDS_20_2.0_LOWER: str = 'BollingerBands_20_2.0_LOWER'
    """
    Feature: Bollinger Bands
        Outputs: lower
        Params: {'period': 20, 'std_dev': 2.0}
    """
    BOLLINGERBANDS_20_2.0_WIDTH: str = 'BollingerBands_20_2.0_WIDTH'
    """
    Feature: Bollinger Bands
        Outputs: width
        Params: {'period': 20, 'std_dev': 2.0}
    """

@dataclass(frozen=True)
class ParamsContext:
    """Strictly typed strategy hyperparameters defined in manifest.json."""
    stop_loss: float = 0.05
    take_profit: float = 0.1

@dataclass(frozen=True)
class Context:
    """Master strategy context object."""
    features: FeaturesContext = field(default_factory=FeaturesContext)
    params: ParamsContext = field(default_factory=ParamsContext)