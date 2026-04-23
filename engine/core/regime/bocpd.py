"""Bayesian Online Change-Point Detection (BOCPD).

Implements Adams & MacKay (2007) with a Normal-Gamma conjugate prior.
The output is a per-bar novelty score: P(run_length < 5 days) at each step.

When this probability exceeds 0.5, the data-generating process has likely
just changed — use RegimeContext.size_multiplier to reduce position sizes.

No external library required; uses only numpy and scipy.
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln


class BayesianCPD:
    """Truncated Bayesian online change-point detector.

    Processes a univariate time series and returns P(run_length < 5) at each
    step, which serves as the novelty score for the regime system.

    The run-length distribution is maintained over a finite window
    (``max_run_length``) to keep memory and compute bounded.

    Args:
        hazard_rate: Probability of a changepoint at any given step.
            Default 1/63 ≈ one changepoint per quarter.
        max_run_length: Maximum run length to track. Excess probability mass
            is absorbed into the last bucket.
        mu0, kappa0, alpha0, beta0: Normal-Gamma prior hyper-parameters.
            Defaults represent weak prior centred at 0 with unit variance.
    """

    def __init__(
        self,
        hazard_rate: float = 1.0 / 63,
        max_run_length: int = 504,
        mu0: float = 0.0,
        kappa0: float = 1.0,
        alpha0: float = 2.0,
        beta0: float = 1.0,
    ):
        self.hazard_rate = hazard_rate
        self.max_run_length = max_run_length
        self.mu0 = mu0
        self.kappa0 = kappa0
        self.alpha0 = alpha0
        self.beta0 = beta0

    def run(self, x: np.ndarray) -> np.ndarray:
        """Compute novelty scores for a univariate time series.

        Args:
            x: 1-D array of observations (e.g. standardised VIX level).

        Returns:
            novelty: 1-D array, same length as ``x``.
                Each value is P(run_length < 5 | x_{1:t}).
        """
        T = len(x)
        H = self.hazard_rate
        mrl = self.max_run_length

        # Sufficient statistics for each run-length hypothesis r = 0 .. mrl
        mu    = np.full(mrl + 1, self.mu0,    dtype=np.float64)
        kappa = np.full(mrl + 1, self.kappa0, dtype=np.float64)
        alpha = np.full(mrl + 1, self.alpha0, dtype=np.float64)
        beta  = np.full(mrl + 1, self.beta0,  dtype=np.float64)

        # Run-length posterior R[r] = P(run_length = r | data so far)
        R = np.zeros(mrl + 1)
        R[0] = 1.0

        novelty = np.zeros(T)

        for t in range(T):
            x_t = float(x[t])
            active = min(t + 1, mrl)   # highest non-zero run-length index

            # ----------------------------------------------------------
            # 1.  Predictive probability: Student-T(dof=2*alpha, loc=mu,
            #     scale=sqrt(beta*(kappa+1)/(alpha*kappa)))
            # ----------------------------------------------------------
            a = alpha[:active + 1]
            k = kappa[:active + 1]
            m = mu[:active + 1]
            b = beta[:active + 1]

            scale2 = b * (k + 1) / (a * k)
            scale2 = np.maximum(scale2, 1e-10)
            dof = 2.0 * a

            z2 = (x_t - m) ** 2 / (dof * scale2)
            log_pred = (
                gammaln((dof + 1) / 2)
                - gammaln(dof / 2)
                - 0.5 * np.log(np.pi * dof * scale2)
                - ((dof + 1) / 2) * np.log1p(z2)
            )
            # Numerical stability: shift by max before exp
            log_pred -= log_pred.max()
            pred = np.exp(log_pred)
            pred = np.maximum(pred, 1e-300)

            # ----------------------------------------------------------
            # 2.  Update run-length posterior
            # ----------------------------------------------------------
            R_active = R[:active + 1]
            R_new = np.zeros(mrl + 1)

            # Growth: run continues — shift by 1, weight by (1-H)*pred
            if active > 0:
                R_new[1:active + 1] = R_active[:active] * pred[:active] * (1.0 - H)

            # Changepoint: any run resets to 0
            R_new[0] = (R_active * pred * H).sum()

            # Normalise
            total = R_new.sum()
            if total > 1e-300:
                R_new /= total
            else:
                R_new[0] = 1.0
            R = R_new

            # Novelty = P(run_length < 5)
            novelty[t] = R[:min(5, active + 2)].sum()

            # ----------------------------------------------------------
            # 3.  Update sufficient statistics for surviving hypotheses
            #     r = 1..active  correspond to old r = 0..active-1 that grew
            # ----------------------------------------------------------
            up_to = min(active, mrl - 1)
            k_old = kappa[:up_to + 1]
            m_old = mu[:up_to + 1]
            a_old = alpha[:up_to + 1]
            b_old = beta[:up_to + 1]

            new_k = k_old + 1.0
            new_m = (k_old * m_old + x_t) / new_k
            new_a = a_old + 0.5
            new_b = b_old + k_old * (x_t - m_old) ** 2 / (2.0 * new_k)

            mu[1:up_to + 2]    = new_m
            kappa[1:up_to + 2] = new_k
            alpha[1:up_to + 2] = new_a
            beta[1:up_to + 2]  = new_b

            # Reset run-length-0 bucket to prior (changepoint hypothesis)
            mu[0]    = self.mu0
            kappa[0] = self.kappa0
            alpha[0] = self.alpha0
            beta[0]  = self.beta0

        return novelty
