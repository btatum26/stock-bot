"""Rule-based regime detectors. No training required; deterministic one-hot output."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import RegimeDetector, register_regime


@register_regime("vix_adx")
class VixAdxRegime(RegimeDetector):
    """3-state regime classifier using VIX level and ADX trend strength.

    State 0 — low-vol trending   : VIX < 15  AND ADX > 25  (favour momentum)
    State 1 — normal ranging     : VIX 15-25 AND ADX < 20  (favour mean-reversion)
    State 2 — stressed           : VIX > 25                (cut size, favour contrarian)

    Output is deterministic one-hot probability vectors, shaped identically to
    probabilistic detectors so downstream code is agnostic to detector type.

    Required macro_features columns: ``vix``, ``adx``.
    """

    @property
    def name(self) -> str:
        return "VixAdxRegime"

    @property
    def n_states(self) -> int:
        return 3

    def fit(self, macro_features: pd.DataFrame) -> None:
        pass

    def predict_proba(self, macro_features: pd.DataFrame) -> pd.DataFrame:
        vix = macro_features["vix"].ffill().fillna(20.0)
        adx = macro_features["adx"].ffill().fillna(20.0)

        n = len(macro_features)
        proba = np.zeros((n, 3))

        stressed      = (vix > 25).values
        low_trending  = ((vix < 15) & (adx > 25)).values
        # everything else → normal ranging (state 1)

        proba[stressed, 2] = 1.0
        proba[~stressed & low_trending, 0] = 1.0
        proba[~stressed & ~low_trending, 1] = 1.0

        return pd.DataFrame(proba, index=macro_features.index, columns=[0, 1, 2])

    def novelty_score(self, macro_features: pd.DataFrame) -> pd.Series:
        return pd.Series(0.0, index=macro_features.index)


@register_regime("term_structure")
class TermStructureRegime(RegimeDetector):
    """2-state regime classifier using the VIX / VIX3M term structure.

    State 0 — calm contango          : ratio < 0.95
    State 1 — stressed backwardation : ratio > 1.00

    The transition band [0.95, 1.00] is filled with a linear blend so the
    output is smooth rather than binary.

    Required macro_features column: ``vix_vix3m``.
    """

    @property
    def name(self) -> str:
        return "TermStructureRegime"

    @property
    def n_states(self) -> int:
        return 2

    def fit(self, macro_features: pd.DataFrame) -> None:
        pass

    def predict_proba(self, macro_features: pd.DataFrame) -> pd.DataFrame:
        ratio = macro_features["vix_vix3m"].ffill().fillna(0.9)

        p_stressed = ((ratio - 0.95) / 0.05).clip(0.0, 1.0)
        p_calm = 1.0 - p_stressed

        return pd.DataFrame(
            {0: p_calm.values, 1: p_stressed.values},
            index=macro_features.index,
        )

    def novelty_score(self, macro_features: pd.DataFrame) -> pd.Series:
        return pd.Series(0.0, index=macro_features.index)
