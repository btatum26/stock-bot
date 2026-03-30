# AUTO-GENERATED FILE. DO NOT MODIFY DIRECTLY.
from dataclasses import dataclass, field

@dataclass(frozen=True)
class FeaturesContext:
    """Strictly typed mapping of strategy features to DataFrame columns."""
    RSI_close_14: str = 'RSI_close_14'
    """
    Feature: RSI
        Outputs: Primary
        Params: {'window': 14, 'source': 'close'}
    """

@dataclass(frozen=True)
class ParamsContext:
    """Strictly typed strategy hyperparameters defined in manifest.json."""
    oversold: float = 30.0
    overbought: float = 70.0
    divergence_lookback: float = 20.0

@dataclass(frozen=True)
class Context:
    """Master strategy context object."""
    features: FeaturesContext = field(default_factory=FeaturesContext)
    params: ParamsContext = field(default_factory=ParamsContext)