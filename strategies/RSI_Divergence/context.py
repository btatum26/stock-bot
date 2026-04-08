# AUTO-GENERATED. Do not edit — updated by the GUI when features change.
from dataclasses import dataclass, field


@dataclass(frozen=True)
class FeaturesContext:
    """Typed mapping from strategy features to DataFrame column names."""
    RSI_14: str = 'RSI_14'
    IS_FRACTAL_HIGH: str = 'Fractals_5_IS_FRACTAL_HIGH'
    IS_FRACTAL_LOW: str = 'Fractals_5_IS_FRACTAL_LOW'
    FRACTAL_HIGH_PRICE: str = 'Fractals_5_FRACTAL_HIGH_PRICE'
    FRACTAL_LOW_PRICE: str = 'Fractals_5_FRACTAL_LOW_PRICE'
    LAST_FRACTAL_HIGH: str = 'Fractals_5_LAST_FRACTAL_HIGH'
    LAST_FRACTAL_LOW: str = 'Fractals_5_LAST_FRACTAL_LOW'
    PREV_FRACTAL_HIGH: str = 'Fractals_5_PREV_FRACTAL_HIGH'
    PREV_FRACTAL_LOW: str = 'Fractals_5_PREV_FRACTAL_LOW'
    STRUCT_HIGH: str = 'Fractals_5_STRUCT_HIGH'
    STRUCT_LOW: str = 'Fractals_5_STRUCT_LOW'
    BARS_SINCE_LAST_HIGH: str = 'Fractals_5_BARS_SINCE_LAST_HIGH'
    BARS_SINCE_LAST_LOW: str = 'Fractals_5_BARS_SINCE_LAST_LOW'
    LAST_SUPPORT_LEVEL: str = 'SupportResistance_0.3_Bill Williams_3.0_0.015_35_LAST_SUPPORT_LEVEL'
    LAST_RESISTANCE_LEVEL: str = 'SupportResistance_0.3_Bill Williams_3.0_0.015_35_LAST_RESISTANCE_LEVEL'


@dataclass(frozen=True)
class ParamsContext:
    """Typed strategy hyperparameters."""
    fractal_n: int = 5
    min_divergence_rsi: float = 5.0
    max_hold_bars: int = 20
    rsi_bull_threshold: float = 55.0
    rsi_bear_threshold: float = 65.0


@dataclass(frozen=True)
class Context:
    """Master context object."""
    features: FeaturesContext = field(default_factory=FeaturesContext)
    params:   ParamsContext   = field(default_factory=ParamsContext)
