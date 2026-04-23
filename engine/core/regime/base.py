"""Abstract base class, registry, and shared data structures for regime detectors."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional, Type

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Global Registry
# ---------------------------------------------------------------------------
REGIME_REGISTRY: Dict[str, Type["RegimeDetector"]] = {}


def register_regime(name: str):
    """Registers a RegimeDetector class into the global registry."""
    def decorator(cls: Type["RegimeDetector"]):
        REGIME_REGISTRY[name] = cls
        return cls
    return decorator


# ---------------------------------------------------------------------------
# Regime Context  (payload passed to generate_signals)
# ---------------------------------------------------------------------------
@dataclass
class RegimeContext:
    """Regime state payload injected into SignalModel.generate_signals.

    Attributes:
        detector_name: Which detector produced this context.
        proba: P(regime=k | x_{1:t}) per bar. Columns are integer state IDs.
        labels: Argmax regime label per bar (int).
        novelty: [0, 1] novelty score per bar. High = structural break / unseen regime.
        n_states: Number of regime states the detector uses.
        ic_weight: Optional per-bar IC weight in [0, 1] from a pre-computed
            conditional IC surface. None if not available.
    """
    detector_name: str
    proba: pd.DataFrame
    labels: pd.Series
    novelty: pd.Series
    n_states: int
    ic_weight: Optional[pd.Series] = None

    def current_regime(self) -> int:
        """Most recent regime label."""
        return int(self.labels.iloc[-1])

    def current_proba(self) -> Dict[int, float]:
        """Regime probability vector at the most recent bar."""
        return self.proba.iloc[-1].to_dict()

    def is_novel(self, threshold: float = 0.5) -> bool:
        """True if the most recent bar's novelty score exceeds the threshold."""
        return float(self.novelty.iloc[-1]) > threshold

    @property
    def size_multiplier(self) -> pd.Series:
        """Position size multiplier from novelty signal.

        Returns 1.0 normally, 0.5 for 10 bars following a novelty spike (> 0.5).
        Multiply raw signals by this before returning from generate_signals.
        """
        novel = (self.novelty > 0.5).values
        result = np.ones(len(novel))
        cooldown = 10
        trigger_until = 0
        for i in range(len(novel)):
            if novel[i]:
                trigger_until = i + cooldown
            if i < trigger_until:
                result[i] = 0.5
        return pd.Series(result, index=self.novelty.index)


# ---------------------------------------------------------------------------
# Abstract Base Class
# ---------------------------------------------------------------------------
class RegimeDetector(ABC):
    """Abstract base class for all regime detection models.

    Subclasses implement fit/predict_proba/novelty_score and register
    themselves with @register_regime("name").
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Display name of this detector."""
        ...

    @property
    @abstractmethod
    def n_states(self) -> int:
        """Number of regime states produced."""
        ...

    @abstractmethod
    def fit(self, macro_features: pd.DataFrame) -> None:
        """Train the detector on macro feature history.

        For rule-based detectors this is a no-op.
        """
        ...

    @abstractmethod
    def predict_proba(self, macro_features: pd.DataFrame) -> pd.DataFrame:
        """Return P(regime=k | features up to t) for every t.

        Must be strictly causal — no lookahead. Columns are int state IDs.
        """
        ...

    @abstractmethod
    def novelty_score(self, macro_features: pd.DataFrame) -> pd.Series:
        """Per-bar novelty: high values indicate an unusual / unseen macro state.

        Rule-based detectors return a constant-zero Series.
        """
        ...
