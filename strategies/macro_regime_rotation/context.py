# AUTO-GENERATED FILE. DO NOT MODIFY DIRECTLY.
from dataclasses import dataclass, field

@dataclass(frozen=True)
class FeaturesContext:
    """Strictly typed mapping of strategy features to DataFrame columns."""
    HYSPREAD_LEVEL: str = 'HYSpread_LEVEL'
    """
    Feature: High-Yield Credit Spread
        Outputs: level
        Params: {}
    """
    HYSPREAD_ROC5: str = 'HYSpread_ROC5'
    """
    Feature: High-Yield Credit Spread
        Outputs: roc5
        Params: {}
    """
    HYSPREAD_ZSCORE: str = 'HYSpread_ZSCORE'
    """
    Feature: High-Yield Credit Spread
        Outputs: zscore
        Params: {}
    """
    NFCI_LEVEL: str = 'NFCI_LEVEL'
    """
    Feature: Chicago Fed Financial Conditions
        Outputs: level
        Params: {}
    """
    NFCI_ROC5: str = 'NFCI_ROC5'
    """
    Feature: Chicago Fed Financial Conditions
        Outputs: roc5
        Params: {}
    """
    NFCI_ZSCORE: str = 'NFCI_ZSCORE'
    """
    Feature: Chicago Fed Financial Conditions
        Outputs: zscore
        Params: {}
    """
    VIXTERMSTRUCTURE: str = 'VIXTermStructure'
    """
    Feature: VIX Term Structure
        Outputs: Primary
        Params: {}
    """
    VIXTERMSTRUCTURE_ZSCORE: str = 'VIXTermStructure_ZSCORE'
    """
    Feature: VIX Term Structure
        Outputs: zscore
        Params: {}
    """

@dataclass(frozen=True)
class ParamsContext:
    """Strictly typed strategy hyperparameters defined in manifest.json."""
    credit_spread_lookback: int = 63
    nfci_lookback_weeks: int = 26
    vix_ratio_bull: float = 0.9
    vix_ratio_bear: float = 1.05

@dataclass(frozen=True)
class Context:
    """Master strategy context object."""
    features: FeaturesContext = field(default_factory=FeaturesContext)
    params: ParamsContext = field(default_factory=ParamsContext)