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
    BOLLINGERBANDS_20_2_0_UPPER: str = 'BollingerBands_20_2.0_UPPER'
    """
    Feature: Bollinger Bands
        Outputs: upper
        Params: {'period': 20, 'std_dev': 2.0}
    """
    BOLLINGERBANDS_20_2_0_MID: str = 'BollingerBands_20_2.0_MID'
    """
    Feature: Bollinger Bands
        Outputs: mid
        Params: {'period': 20, 'std_dev': 2.0}
    """
    BOLLINGERBANDS_20_2_0_LOWER: str = 'BollingerBands_20_2.0_LOWER'
    """
    Feature: Bollinger Bands
        Outputs: lower
        Params: {'period': 20, 'std_dev': 2.0}
    """
    BOLLINGERBANDS_20_2_0_WIDTH: str = 'BollingerBands_20_2.0_WIDTH'
    """
    Feature: Bollinger Bands
        Outputs: width
        Params: {'period': 20, 'std_dev': 2.0}
    """
    AVERAGETRUERANGE_14: str = 'AverageTrueRange_14'
    """
    Feature: ATR
        Outputs: Primary
        Params: {'period': 14, 'normalize': 'none'}
    """
    AVERAGETRUERANGE_14: str = 'Norm_AverageTrueRange_14'
    """
    Feature: ATR
        Outputs: Primary
        Params: {'period': 14, 'normalize': 'pct_distance'}
    """
    ADX_14: str = 'ADX_14'
    """
    Feature: ADX
        Outputs: Primary
        Params: {'period': 14}
    """
    ADX_14_PLUS_DI: str = 'ADX_14_PLUS_DI'
    """
    Feature: ADX
        Outputs: plus_di
        Params: {'period': 14}
    """
    ADX_14_MINUS_DI: str = 'ADX_14_MINUS_DI'
    """
    Feature: ADX
        Outputs: minus_di
        Params: {'period': 14}
    """
    MOVINGAVERAGE_20_EMA: str = 'MovingAverage_20_EMA'
    """
    Feature: Moving Average
        Outputs: Primary
        Params: {'period': 20, 'type': 'EMA', 'normalize': 'none'}
    """
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
    MOVINGAVERAGE_50_EMA: str = 'Norm_MovingAverage_50_EMA'
    """
    Feature: Moving Average
        Outputs: Primary
        Params: {'period': 50, 'type': 'EMA', 'normalize': 'pct_distance'}
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
    stop_loss: float = 0.07
    n_estimators: int = 300
    max_depth: int = 4
    learning_rate: float = 0.05
    min_child_weight: int = 50
    subsample: float = 0.7
    colsample_bytree: float = 0.7
    reg_lambda: float = 2.0
    reg_alpha: float = 0.1
    gamma: float = 0.1
    lookforward: int = 20
    max_drawdown_threshold: float = 0.05
    min_profit_threshold: float = 0.03
    low_vol_threshold: float = 0.25
    high_vol_threshold: float = 0.75
    atr_lookback: int = 252
    vol_expansion_limit: float = 0.1
    entry_quality_threshold: float = 0.35
    min_adx_entry: int = 20
    max_rsi_entry: int = 75
    max_bb_pct_b_entry: float = 0.95
    resistance_buffer_pct: float = 0.02
    trailing_stop_pct: float = 0.1
    max_hold_days: int = 60

@dataclass(frozen=True)
class Context:
    """Master strategy context object."""
    features: FeaturesContext = field(default_factory=FeaturesContext)
    params: ParamsContext = field(default_factory=ParamsContext)