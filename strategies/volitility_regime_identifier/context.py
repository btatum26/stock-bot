# AUTO-GENERATED FILE. DO NOT MODIFY DIRECTLY.
from dataclasses import dataclass, field

@dataclass(frozen=True)
class FeaturesContext:
    """Strictly typed mapping of strategy features to DataFrame columns."""
    MOVINGAVERAGE_50_EMA: str = 'MovingAverage_50_EMA'
    """
    Feature: Moving Average
        Outputs: Primary
        Params: {'period': 50, 'type': 'EMA', 'normalize': 'none'}
    """
    MOVINGAVERAGE_200_EMA: str = 'MovingAverage_200_EMA'
    """
    Feature: Moving Average
        Outputs: Primary
        Params: {'period': 200, 'type': 'EMA', 'normalize': 'none'}
    """
    AVERAGETRUERANGE_14: str = 'AverageTrueRange_14'
    """
    Feature: ATR
        Outputs: Primary
        Params: {'period': 14, 'normalize': 'none'}
    """
    YEARLYCYCLE_SIN: str = 'YearlyCycle_SIN'
    """
    Feature: Yearly Cycle
        Outputs: sin
        Params: {}
    """
    YEARLYCYCLE_COS: str = 'YearlyCycle_COS'
    """
    Feature: Yearly Cycle
        Outputs: cos
        Params: {}
    """
    SUPPORTRESISTANCE_0_02_Bill_Williams_1_0_0_015_35_NEAREST_SUPPORT_LEVEL: str = 'SupportResistance_0.02_Bill Williams_1.0_0.015_35_NEAREST_SUPPORT_LEVEL'
    """
    Feature: Support & Resistance
        Outputs: nearest_support_level
        Params: {'method': 'Bill Williams', 'threshold_pct': 0.015, 'window': 35, 'clustering_pct': 0.02, 'min_strength': 1.0}
    """
    SUPPORTRESISTANCE_0_02_Bill_Williams_1_0_0_015_35_SECOND_SUPPORT_LEVEL: str = 'SupportResistance_0.02_Bill Williams_1.0_0.015_35_SECOND_SUPPORT_LEVEL'
    """
    Feature: Support & Resistance
        Outputs: second_support_level
        Params: {'method': 'Bill Williams', 'threshold_pct': 0.015, 'window': 35, 'clustering_pct': 0.02, 'min_strength': 1.0}
    """
    SUPPORTRESISTANCE_0_02_Bill_Williams_1_0_0_015_35_NEAREST_RESISTANCE_LEVEL: str = 'SupportResistance_0.02_Bill Williams_1.0_0.015_35_NEAREST_RESISTANCE_LEVEL'
    """
    Feature: Support & Resistance
        Outputs: nearest_resistance_level
        Params: {'method': 'Bill Williams', 'threshold_pct': 0.015, 'window': 35, 'clustering_pct': 0.02, 'min_strength': 1.0}
    """
    SUPPORTRESISTANCE_0_02_Bill_Williams_1_0_0_015_35_SECOND_RESISTANCE_LEVEL: str = 'SupportResistance_0.02_Bill Williams_1.0_0.015_35_SECOND_RESISTANCE_LEVEL'
    """
    Feature: Support & Resistance
        Outputs: second_resistance_level
        Params: {'method': 'Bill Williams', 'threshold_pct': 0.015, 'window': 35, 'clustering_pct': 0.02, 'min_strength': 1.0}
    """
    SUPPORTRESISTANCE_0_02_Bill_Williams_1_0_0_015_35_BREAKOUT_RESISTANCE: str = 'SupportResistance_0.02_Bill Williams_1.0_0.015_35_BREAKOUT_RESISTANCE'
    """
    Feature: Support & Resistance
        Outputs: breakout_resistance
        Params: {'method': 'Bill Williams', 'threshold_pct': 0.015, 'window': 35, 'clustering_pct': 0.02, 'min_strength': 1.0}
    """
    SUPPORTRESISTANCE_0_02_Bill_Williams_1_0_0_015_35_BREAKDOWN_SUPPORT: str = 'SupportResistance_0.02_Bill Williams_1.0_0.015_35_BREAKDOWN_SUPPORT'
    """
    Feature: Support & Resistance
        Outputs: breakdown_support
        Params: {'method': 'Bill Williams', 'threshold_pct': 0.015, 'window': 35, 'clustering_pct': 0.02, 'min_strength': 1.0}
    """

@dataclass(frozen=True)
class ParamsContext:
    """Strictly typed strategy hyperparameters defined in manifest.json."""
    low_vol_threshold: float = 0.25
    high_vol_threshold: float = 0.8
    atr_lookback: int = 300
    atr_roc_max: float = 0.1
    stop_loss: float = 0.07

@dataclass(frozen=True)
class Context:
    """Master strategy context object."""
    features: FeaturesContext = field(default_factory=FeaturesContext)
    params: ParamsContext = field(default_factory=ParamsContext)