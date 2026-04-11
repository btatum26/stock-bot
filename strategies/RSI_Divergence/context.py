# AUTO-GENERATED. Do not edit — updated by the GUI when features change.
from dataclasses import dataclass, field


@dataclass(frozen=True)
class FeaturesContext:
    """Typed mapping from strategy features to DataFrame column names."""
    RSI_14: str = 'RSI_14'
    FRACTAL_HIGH_PRICE: str = 'Fractals_4_FRACTAL_HIGH_PRICE'
    FRACTAL_LOW_PRICE: str = 'Fractals_4_FRACTAL_LOW_PRICE'
    LAST_FRACTAL_HIGH: str = 'Fractals_4_LAST_FRACTAL_HIGH'
    LAST_FRACTAL_LOW: str = 'Fractals_4_LAST_FRACTAL_LOW'
    PREV_FRACTAL_HIGH: str = 'Fractals_4_PREV_FRACTAL_HIGH'
    PREV_FRACTAL_LOW: str = 'Fractals_4_PREV_FRACTAL_LOW'
    STRUCT_HIGH: str = 'Fractals_4_STRUCT_HIGH'
    STRUCT_LOW: str = 'Fractals_4_STRUCT_LOW'
    BARS_SINCE_LAST_HIGH: str = 'Fractals_4_BARS_SINCE_LAST_HIGH'
    BARS_SINCE_LAST_LOW: str = 'Fractals_4_BARS_SINCE_LAST_LOW'
    LAST_SUPPORT_LEVEL: str = 'SupportResistance_0.3_Bill Williams_3.0_0.015_35_LAST_SUPPORT_LEVEL'
    LAST_RESISTANCE_LEVEL: str = 'SupportResistance_0.3_Bill Williams_3.0_0.015_35_LAST_RESISTANCE_LEVEL'
    NEAREST_SUPPORT_LEVEL: str = 'SupportResistance_0.3_Bill Williams_3.0_0.015_35_NEAREST_SUPPORT_LEVEL'
    NEAREST_RESISTANCE_LEVEL: str = 'SupportResistance_0.3_Bill Williams_3.0_0.015_35_NEAREST_RESISTANCE_LEVEL'


@dataclass(frozen=True)
class ParamsContext:
    """Typed strategy hyperparameters."""
    low_vol_threshold: float = 0.25
    high_vol_threshold: float = 0.75
    atr_lookback: int = 252
    atr_roc_max: float = 0.1
    stop_loss: float = 0.07


@dataclass(frozen=True)
class Context:
    """Master context object."""
    features: FeaturesContext = field(default_factory=FeaturesContext)
    params:   ParamsContext   = field(default_factory=ParamsContext)
