"""Deflated Sharpe Ratio — Bailey & López de Prado (2014).

DSR adjusts the observed Sharpe ratio for:
  (a) Non-normal return distribution (skewness + kurtosis via Mertens 2002)
  (b) Multiple trials — expected maximum SR under H0 grows with trial count

Returns a probability in [0, 1]:
  DSR > 0.95  strong evidence of genuine edge
  DSR > 0.50  passes the basic bar (more likely signal than luck)
  DSR < 0.50  looks like lucky noise given the number of trials run

No external dependencies beyond scipy (already a project dependency).
"""
import math

import numpy as np
from scipy.stats import norm

_EULER_MASCHERONI = 0.5772156649015329


def sr_star(n_trials: int, n_obs: int) -> float:
    """Expected maximum per-period SR under H0 across n_trials independent tests.

    Under the null hypothesis (no edge), if you run n_trials independent
    strategies each evaluated on n_obs observations, the expected Sharpe of
    the best one follows this approximation (Bailey & de Prado 2014, eq. 8).
    Returns 0.0 for n_trials <= 1 (no benchmark inflation for a single test).
    """
    if n_trials <= 1:
        return 0.0
    n_trials = max(n_trials, 2)
    n_obs = max(n_obs, 2)
    p1 = (1.0 - _EULER_MASCHERONI) * norm.ppf(1.0 - 1.0 / n_trials)
    p2 = _EULER_MASCHERONI * norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return (p1 + p2) / math.sqrt(n_obs - 1)


def compute_dsr(returns, n_trials: int = 1) -> float:
    """Deflated Sharpe Ratio (probability value in [0, 1]).

    Parameters
    ----------
    returns  : array-like of per-period strategy returns (daily for 1d bars).
               NaN values are dropped before computation.
    n_trials : total number of backtests + training runs ever executed for
               this strategy. Obtained from trial_counter.get_total_trials().

    Returns
    -------
    float in [0, 1].  NaN if data is insufficient or inputs are degenerate.
    """
    import pandas as pd

    s = pd.Series(returns).dropna()
    n_obs = len(s)
    if n_obs < 20:
        return float("nan")

    mu = float(s.mean())
    sigma = float(s.std(ddof=1))
    if sigma == 0.0 or not math.isfinite(sigma):
        return float("nan")

    sr = mu / sigma  # per-period (not annualised)

    skew = float(s.skew())              # Fisher third standardised moment
    excess_kurt = float(s.kurtosis())   # Fisher excess kurtosis (pandas default)
    raw_kurt = excess_kurt + 3.0        # raw kurtosis used in Mertens (2002) formula

    sr_bench = sr_star(n_trials, n_obs)

    # Mertens (2002) variance of the Sharpe estimator under non-normality
    var_term = 1.0 - skew * sr + (raw_kurt - 1.0) / 4.0 * sr ** 2
    if var_term <= 0.0:
        return float("nan")

    z = (sr - sr_bench) * math.sqrt(n_obs - 1) / math.sqrt(var_term)
    if not math.isfinite(z):
        return float("nan")

    return float(norm.cdf(z))
