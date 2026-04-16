"""Rule-based entry and exit confirmations.

Entry confirmations gate ML-suggested entries with structural checks.
Exit conditions are purely rule-based and cannot be overridden by ML.

All functions operate on raw OHLCV + pre-computed feature columns passed
via the full DataFrame in generate_signals.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class ConfirmationResult:
    """Aggregate result of rule-based confirmation checks."""

    passed: bool
    checks: Dict[str, bool]
    reason: Optional[str] = None

    @classmethod
    def fail(cls, reason: str, checks: dict) -> "ConfirmationResult":
        return cls(passed=False, checks=checks, reason=reason)

    @classmethod
    def success(cls, checks: dict) -> "ConfirmationResult":
        return cls(passed=True, checks=checks)


# ---------------------------------------------------------------------------
# Individual entry checks
# ---------------------------------------------------------------------------

def check_trend(
    close: float,
    ema_50: float,
    ema_200: float,
    adx: float,
    params: dict,
) -> bool:
    """EMA alignment + minimum trend strength."""
    min_adx = float(params.get("min_adx_entry", 20))
    return (close > ema_50) and (ema_50 > ema_200) and (adx >= min_adx)


def check_volatility(
    atr_pct: float,
    atr_roc: float,
    params: dict,
) -> bool:
    """ATR percentile below high-vol threshold and vol not expanding."""
    high_vol_thr = float(params.get("high_vol_threshold", 0.75))
    vol_limit = float(params.get("vol_expansion_limit", 0.10))
    if np.isnan(atr_pct) or np.isnan(atr_roc):
        return False
    return (atr_pct < high_vol_thr) and (atr_roc < vol_limit)


def check_not_overbought(
    rsi: float,
    bb_pct_b: float,
    params: dict,
) -> bool:
    """Reject entries in extremely overbought territory."""
    max_rsi = float(params.get("max_rsi_entry", 75))
    max_bb = float(params.get("max_bb_pct_b_entry", 0.95))
    if np.isnan(rsi):
        return True  # passthrough when indicator unavailable
    rsi_ok = rsi < max_rsi
    bb_ok = np.isnan(bb_pct_b) or (bb_pct_b < max_bb)
    return rsi_ok and bb_ok


def check_not_at_resistance(
    close: float,
    nearest_resistance: float,
    params: dict,
) -> bool:
    """Reject entries within buffer of nearest resistance."""
    buf = float(params.get("resistance_buffer_pct", 0.02))
    if np.isnan(nearest_resistance) or nearest_resistance <= close:
        return True  # no overhead resistance -> pass
    if close == 0:
        return True
    dist = (nearest_resistance - close) / close
    return dist > buf


# ---------------------------------------------------------------------------
# Combined entry confirmation
# ---------------------------------------------------------------------------

def check_entry_confirmations(
    close: float,
    ema_50: float,
    ema_200: float,
    adx: float,
    atr_pct: float,
    atr_roc: float,
    rsi: float,
    bb_pct_b: float,
    nearest_resistance: float,
    params: dict,
) -> ConfirmationResult:
    """Run all entry confirmations and return aggregate result."""
    checks: Dict[str, bool] = {}

    checks["trend"] = check_trend(close, ema_50, ema_200, adx, params)
    checks["volatility"] = check_volatility(atr_pct, atr_roc, params)
    checks["not_overbought"] = check_not_overbought(rsi, bb_pct_b, params)
    checks["not_at_resistance"] = check_not_at_resistance(
        close, nearest_resistance, params
    )

    required = ["trend", "volatility", "not_overbought", "not_at_resistance"]
    all_pass = all(checks[c] for c in required)

    if not all_pass:
        failed = [c for c in required if not checks[c]]
        return ConfirmationResult.fail(
            reason=f"Failed: {failed}", checks=checks
        )
    return ConfirmationResult.success(checks=checks)


# ---------------------------------------------------------------------------
# Exit conditions
# ---------------------------------------------------------------------------

def check_exit_conditions(
    close: float,
    ema_50: float,
    entry_price: float,
    peak_since_entry: float,
    current_regime: int,
    params: dict,
) -> Tuple[bool, str]:
    """Check if any exit condition fires.

    Returns:
        (should_exit, reason) tuple.
    """
    # 1. Regime exit: RISK_OFF
    if current_regime == 0:
        return True, "regime_risk_off"

    # 2. Trend break: close below EMA_50
    if not np.isnan(ema_50) and close < ema_50:
        return True, "trend_break_ema50"

    # 3. Hard stop loss
    stop_pct = float(params.get("stop_loss", 0.07))
    pnl = (close - entry_price) / entry_price
    if pnl < -stop_pct:
        return True, "stop_loss"

    # 4. Trailing stop (percentage from peak)
    trail_pct = float(params.get("trailing_stop_pct", 0.10))
    if peak_since_entry > 0:
        dd_from_peak = (peak_since_entry - close) / peak_since_entry
        if dd_from_peak > trail_pct:
            return True, "trailing_stop"

    # 5. Time stop
    max_hold = int(params.get("max_hold_days", 60))
    # (caller tracks bars_held and passes as param if needed;
    #  kept simple here — caller should check externally)

    return False, ""
