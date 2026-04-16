# AUTO-GENERATED. Do not edit — updated by the GUI when features change.
from dataclasses import dataclass, field


@dataclass(frozen=True)
class FeaturesContext:
    """Typed mapping from strategy features to DataFrame column names."""
    RSI_14: str = 'RSI_14'
    UPPER: str = 'BollingerBands_20_2.0_UPPER'
    MID: str = 'BollingerBands_20_2.0_MID'
    LOWER: str = 'BollingerBands_20_2.0_LOWER'
    WIDTH: str = 'BollingerBands_20_2.0_WIDTH'
    AverageTrueRange_14: str = 'AverageTrueRange_14'
    Norm_AverageTrueRange_14: str = 'Norm_AverageTrueRange_14'
    ADX_14: str = 'ADX_14'
    PLUS_DI: str = 'ADX_14_PLUS_DI'
    MINUS_DI: str = 'ADX_14_MINUS_DI'
    MovingAverage_20_EMA: str = 'MovingAverage_20_EMA'
    MovingAverage_50_EMA: str = 'MovingAverage_50_EMA'
    MovingAverage_200_EMA: str = 'MovingAverage_200_EMA'
    Norm_MovingAverage_50_EMA: str = 'Norm_MovingAverage_50_EMA'
    SIN: str = 'YearlyCycle_SIN'
    COS: str = 'YearlyCycle_COS'
    NEAREST_SUPPORT_LEVEL: str = 'SupportResistance_0.02_Bill Williams_1.0_0.015_35_NEAREST_SUPPORT_LEVEL'
    SECOND_SUPPORT_LEVEL: str = 'SupportResistance_0.02_Bill Williams_1.0_0.015_35_SECOND_SUPPORT_LEVEL'
    NEAREST_RESISTANCE_LEVEL: str = 'SupportResistance_0.02_Bill Williams_1.0_0.015_35_NEAREST_RESISTANCE_LEVEL'
    SECOND_RESISTANCE_LEVEL: str = 'SupportResistance_0.02_Bill Williams_1.0_0.015_35_SECOND_RESISTANCE_LEVEL'
    BREAKOUT_RESISTANCE: str = 'SupportResistance_0.02_Bill Williams_1.0_0.015_35_BREAKOUT_RESISTANCE'
    BREAKDOWN_SUPPORT: str = 'SupportResistance_0.02_Bill Williams_1.0_0.015_35_BREAKDOWN_SUPPORT'
    RSI_7: str = 'RSI_7'
    ROC_20: str = 'ROC_20'
    Norm_MovingAverage_252_SMA: str = 'Norm_MovingAverage_252_SMA'
    OBV: str = 'OBV'
    SMA_20: str = 'OBV_SMA_20'


@dataclass(frozen=True)
class ParamsContext:
    """Typed strategy hyperparameters."""
    stop_loss: float = 0.05
    n_estimators: int = 200
    max_depth: int = 3
    learning_rate: float = 0.04
    min_child_weight: int = 10
    subsample: float = 0.7
    colsample_bytree: float = 0.7
    reg_lambda: float = 2.0
    reg_alpha: float = 0.1
    gamma: float = 0.1
    lookforward: int = 20
    max_drawdown_threshold: float = 0.05
    min_profit_threshold: float = 0.03
    low_vol_threshold: float = 0.3
    high_vol_threshold: float = 0.75
    atr_lookback: int = 246
    vol_expansion_limit: float = 0.1
    entry_quality_threshold: float = 0.34999999999999964
    min_adx_entry: int = 15
    max_rsi_entry: int = 72
    max_bb_pct_b_entry: float = 0.8
    resistance_buffer_pct: float = 0.02
    trailing_stop_pct: float = 0.05
    max_hold_days: int = 200


@dataclass(frozen=True)
class Context:
    """Master context object."""
    features: FeaturesContext = field(default_factory=FeaturesContext)
    params:   ParamsContext   = field(default_factory=ParamsContext)
