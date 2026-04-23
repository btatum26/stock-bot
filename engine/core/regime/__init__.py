"""Regime detection subsystem.

Import order matters: base defines the registry; detectors register on import.
"""
from .base import (
    REGIME_REGISTRY,
    RegimeContext,
    RegimeDetector,
    register_regime,
)
from .rule_based import VixAdxRegime, TermStructureRegime  # noqa: F401 — registers on import
from .hmm import GaussianHMMRegime                         # noqa: F401
from .bocpd import BayesianCPD
from .orchestrator import RegimeOrchestrator

__all__ = [
    "REGIME_REGISTRY",
    "RegimeContext",
    "RegimeDetector",
    "register_regime",
    "VixAdxRegime",
    "TermStructureRegime",
    "GaussianHMMRegime",
    "BayesianCPD",
    "RegimeOrchestrator",
]
