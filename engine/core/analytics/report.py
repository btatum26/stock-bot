"""Standardized IC evaluation report renderer."""
from __future__ import annotations

import math
from typing import Optional

from .ic_analyzer import ICResult
from .conditional_ic import ConditionalICResult


_HORIZONS = [1, 5, 10, 20, 60]
_W = 64


def render_ic_report(
    result: ICResult,
    dsr: Optional[float] = None,
) -> str:
    lines: list[str] = []

    def w(s: str = "") -> None:
        lines.append(s)

    def _f(v: float, fmt: str = ".4f") -> str:
        return format(v, fmt) if math.isfinite(v) else "N/A"

    w("=" * _W)
    w(f"SIGNAL: {result.signal_name}")
    universe_str = ", ".join(result.universe[:10])
    if len(result.universe) > 10:
        universe_str += f" (+{len(result.universe) - 10} more)"
    w(f"UNIVERSE: {universe_str}  ({len(result.universe)} tickers)")
    w(f"PERIOD: {result.period_start} -> {result.period_end}")
    w(f"TRIALS TESTED BEFORE THIS ONE: {result.n_trials}")
    w()

    if len(result.universe) < 10:
        w(f"  WARNING: Only {len(result.universe)} tickers. Cross-sectional IC estimates")
        w(f"  are unreliable below ~20. Consider a broader universe.")
        w()

    w("INFORMATION COEFFICIENT")
    for h in _HORIZONS:
        ic = result.mean_ic.get(h, float("nan"))
        ir = result.ic_ir.get(h, float("nan"))
        pct = result.ic_pos_frac.get(h, float("nan"))
        pct_s = f"{pct:.1%}" if math.isfinite(pct) else "N/A"
        suffix = ""
        if h == 5:
            if math.isfinite(ic) and ic >= 0.05:
                suffix = "  [strong]"
            elif math.isfinite(ic) and ic >= 0.02:
                suffix = "  [detectable]"
            elif math.isfinite(ic):
                suffix = "  [below threshold]"
        w(f"  {h:2d}-day:  IC = {_f(ic)},  IC-IR = {_f(ir)},  pct positive = {pct_s}{suffix}")

    # Decay pattern
    finite_pairs = [(h, result.mean_ic[h]) for h in _HORIZONS
                    if h in result.mean_ic and math.isfinite(result.mean_ic[h])]
    if len(finite_pairs) >= 3:
        vals = [v for _, v in finite_pairs]
        diffs = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
        n_neg = sum(1 for d in diffs if d < 0)
        if n_neg == len(diffs):
            decay = "smooth decay"
        elif n_neg == 0:
            decay = "flat or rising (unusual)"
        else:
            decay = "erratic"
    else:
        decay = "insufficient data"
    w(f"  Decay pattern: {decay}")
    w()

    w("QUINTILE RETURNS (annualized, 5-day horizon)")
    q_df = result.quintile_returns
    if q_df is not None and 5 in q_df.index:
        row = q_df.loc[5]
        vals = {q: row.get(q, float("nan")) for q in range(1, 6)}
        q_line = "  ".join(
            f"Q{q}={_f(vals[q], '+.2%')}" if math.isfinite(vals[q]) else f"Q{q}=N/A"
            for q in range(1, 6)
        )
        w(f"  {q_line}")
        spread = result.quintile_spread.get(5, float("nan"))
        mono = result.quintile_monotonic.get(5, False)
        w(f"  Q5-Q1 spread: {_f(spread, '+.2%') if math.isfinite(spread) else 'N/A'}")
        w(f"  Monotonic? {'Y' if mono else 'N'}")
    else:
        w("  N/A")
    w()

    w("TURNOVER")
    to = result.daily_turnover
    hl = result.half_life_days
    w(f"  Daily: {_f(to, '.1%') if math.isfinite(to) else 'N/A'}")
    w(f"  Half-life: {_f(hl, '.0f') + ' days' if math.isfinite(hl) else 'N/A'}")
    if math.isfinite(to) and to > 0.50:
        w("  WARNING: Turnover > 50%/day — signal cannot survive realistic cost models.")
    w()

    w("ROBUSTNESS")
    pc = result.period_ic_consistency
    w(f"  Period IC consistency (first/second half ratio): {_f(pc)}")
    w()

    if dsr is not None and math.isfinite(dsr):
        w(f"DEFLATED SHARPE (after {result.n_trials} trial adjustment): {dsr:.4f}")
        w()

    # Verdict
    ic5 = result.mean_ic.get(5, float("nan"))
    ir5 = result.ic_ir.get(5, float("nan"))
    if not (math.isfinite(ic5) and math.isfinite(ir5)):
        verdict = "DISCARD  (insufficient data to evaluate)"
    elif ic5 >= 0.05 and ir5 >= 0.5:
        verdict = "PROCEED -> CONDITIONAL IC SURFACE  (strong edge)"
    elif ic5 >= 0.02 and ir5 >= 0.3:
        verdict = "PROCEED -> CONDITIONAL IC SURFACE  (weak but detectable edge)"
    else:
        verdict = f"DISCARD  (IC={ic5:.4f} < 0.02 or IC-IR={ir5:.4f} < 0.3)"

    w(f"VERDICT: {verdict}")
    w("=" * _W)

    return "\n".join(lines)


def render_conditional_ic_report(
    result: ConditionalICResult,
    signal_name: str = "signal",
) -> str:
    lines: list[str] = []

    def w(s: str = "") -> None:
        lines.append(s)

    def _f(v: float, fmt: str = ".4f") -> str:
        return format(v, fmt) if math.isfinite(v) else "N/A"

    w("=" * _W)
    w(f"CONDITIONAL IC SURFACE: {signal_name}")
    dims_str = "  x  ".join(f"{s.series_id} ({s.source}, {s.n_bins} bins)" for s in result.dims)
    w(f"MACRO DIMENSIONS: {dims_str}")
    w(f"HORIZON: {result.horizon}d")
    w(f"UNCONDITIONAL IC: {_f(result.unconditional_ic)}")
    w()

    if not result.bins:
        w("  No regime bins met the minimum observation threshold.")
        w(f"  DIAGNOSIS: {result.diagnosis.upper()}")
        w(f"  {result.diagnosis_detail}")
        w("=" * _W)
        return "\n".join(lines)

    col_w = max(len(b.label) for b in result.bins)
    col_w = min(max(col_w, 30), 56)

    w("REGIME IC TABLE  (sorted by mean IC, descending)")
    w(f"  {'Regime':<{col_w}}  {'IC':>8}  {'IC-IR':>7}  {'N':>6}")
    w(f"  {'-' * (col_w + 26)}")
    for b in result.bins:
        marker = " *" if b.mean_ic >= result.unconditional_ic + 0.02 else "  "
        w(f"  {b.label:<{col_w}}  {_f(b.mean_ic):>8}  {_f(b.ic_ir):>7}  {b.n_obs:>6}{marker}")
    w("  (* = IC meaningfully above unconditional)")
    w()

    w(f"DIAGNOSIS: {result.diagnosis.upper()}")
    w(f"  {result.diagnosis_detail}")
    w()

    if result.diagnosis == "structured":
        best = result.bins[0]
        w("RECOMMENDED ACTION:")
        w(f"  Best regime: {best.label}")
        w(f"  IC in regime = {_f(best.mean_ic)}, vs unconditional = {_f(result.unconditional_ic)}")
        w("  In model.py: scale signal by (regime_ic / unconditional_ic) when in this regime.")
        w("  Use current macro values at signal-generation time to identify regime.")
    elif result.diagnosis == "flat":
        w("RECOMMENDED ACTION:")
        w("  No regime filter needed. Deploy signal uniformly across all market conditions.")
    else:
        w("RECOMMENDED ACTION:")
        w("  Investigate alternative macro dimensions, or discard this signal.")

    w("=" * _W)
    return "\n".join(lines)
