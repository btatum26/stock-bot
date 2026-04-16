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
    MOVINGAVERAGE_50_SMA: str = 'MovingAverage_50_SMA'
    """
    Feature: Moving Average
        Outputs: Primary
        Params: {'period': 50, 'type': 'SMA'}
    """
    MOVINGAVERAGE_200_SMA: str = 'MovingAverage_200_SMA'
    """
    Feature: Moving Average
        Outputs: Primary
        Params: {'period': 200, 'type': 'SMA'}
    """
    VOLUME: str = 'Volume'
    """
    Feature: Volume
        Outputs: Primary
        Params: {}
    """
    TICKER_COMPARE_SPY_200_RATIO: str = 'TICKER_COMPARE_SPY_200_RATIO'
    """
    Feature: Ticker Comparison
        Outputs: ratio
        Params: {'compare_ticker': 'SPY', 'window': 200}
    """
    TICKER_COMPARE_SPY_200_CORR: str = 'TICKER_COMPARE_SPY_200_CORR'
    """
    Feature: Ticker Comparison
        Outputs: corr
        Params: {'compare_ticker': 'SPY', 'window': 200}
    """
    TICKER_COMPARE_SPY_200_BETA: str = 'TICKER_COMPARE_SPY_200_BETA'
    """
    Feature: Ticker Comparison
        Outputs: beta
        Params: {'compare_ticker': 'SPY', 'window': 200}
    """
    TICKER_COMPARE_SPY_200_REGIME: str = 'TICKER_COMPARE_SPY_200_REGIME'
    """
    Feature: Ticker Comparison
        Outputs: regime
        Params: {'compare_ticker': 'SPY', 'window': 200}
    """

@dataclass(frozen=True)
class ParamsContext:
    """Strictly typed strategy hyperparameters defined in manifest.json."""
    stop_loss: float = 0.05
    take_profit: float = 0.1
    min_consol_days: int = 10
    max_consol_days: int = 100
    range_pct_threshold: float = 0.15
    atr_contract_thresh: float = 0.9
    breakout_margin: float = 0.05
    volume_surge: float = 1.1
    require_volume: int = 1
    use_regime_filter: int = 1
    base_atr_mult: float = 1.5
    length_scale: float = 0.025
    max_atr_mult: float = 4.0
    min_atr_mult: float = 1.5
    ref_length: int = 15
    exit_on_close: int = 1
    max_hold_days: int = 120
    failed_breakout_days: int = 5

@dataclass(frozen=True)
class Context:
    """Master strategy context object."""
    features: FeaturesContext = field(default_factory=FeaturesContext)
    params: ParamsContext = field(default_factory=ParamsContext)