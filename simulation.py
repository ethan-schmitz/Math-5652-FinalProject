"""
simulation.py

Core engine for the Compound Poisson aggregate loss model.

    S(t) = sum_{i=1}^{N(t)} X_i

where:
  N(t) ~ Poisson(lambda * t)  -- number of claims in [0, t]
  X_i  ~ severity distribution -- i.i.d. claim sizes, independent of N

This is the standard actuarial frequency-severity framework. It produces
a full DISTRIBUTION of total losses rather than a single point estimate.
"""

import numpy as np
from typing import Callable, Optional

def simulate_aggregate_loss(
    lam: float,
    t: float,
    severity_fn: Callable[[int, np.random.Generator], np.ndarray],
    rng: np.random.Generator,
) -> float:
    """
    Simulate one realisation of S(t).

    Algorithm:
      1. Draw  N ~ Poisson(lambda * t) - claim COUNT
      2. If N = 0, return 0.0 immediately
      3. Draw  X_1, ..., X_N  from severity_fn - claim SIZES
      4. Return  S(t) = X_1 + ... + X_N

    Parameters
    lam: Poisson arrival rate (claims per unit time).
    t: Observation window length.
    severity_fn: (n, rng) -> ndarray of shape (n,). See distributions.py.
    rng: Shared numpy Generator for reproducibility.

    Returns
    float : S(t) for this single path.
    """
    n_claims: int = rng.poisson(lam * t)  # Step 1: how many claims?

    if n_claims == 0:                      # Step 2: no claims → zero loss
        return 0.0

    severities = severity_fn(n_claims, rng)  # Step 3: draw claim sizes
    return float(np.sum(severities))          # Step 4: sum them up


def monte_carlo_simulation(
    lam: float,
    t: float,
    severity_fn: Callable[[int, np.random.Generator], np.ndarray],
    n_simulations: int = 50_000,
    seed: Optional[int] = 42,
) -> np.ndarray:
    """
    Run n_simulations independent realisations of S(t).

    By the Law of Large Numbers, the empirical distribution converges to
    the true distribution of S(t) as n_simulations → ∞.

    Parameters
    lam: Poisson arrival rate.
    t: Observation window.
    severity_fn: Severity sampling function (distributions.py).
    n_simulations: Number of Monte Carlo replications.
    seed: Random seed (None = non-reproducible).

    Returns
    np.ndarray of shape (n_simulations,)
    """
    rng = np.random.default_rng(seed)
    losses = np.empty(n_simulations, dtype=float)
    for i in range(n_simulations):
        losses[i] = simulate_aggregate_loss(lam, t, severity_fn, rng)
    return losses


def parameter_sweep(
    base_lam: float,
    base_t: float,
    severity_fn: Callable[[int, np.random.Generator], np.ndarray],
    param_name: str,
    param_values: list,
    n_simulations: int = 20_000,
    seed: int = 42,
) -> dict:
    """
    Hold all parameters constant except one, sweep that one across
    param_values, and return a dict of {value: loss_array}.

    This is how actuaries do sensitivity analysis — understanding which
    model inputs drive risk the most.

    Parameters
    param_name: 'lambda' to sweep arrival rate, 't' to sweep time.
    param_values: e.g. [5, 10, 20, 50]

    Returns
    dict { param_value : np.ndarray of shape (n_simulations,) }
    """
    if param_name not in ("lambda", "t"):
        raise ValueError("param_name must be 'lambda' or 't'.")

    results = {}
    for val in param_values:
        lam_use = val if param_name == "lambda" else base_lam
        t_use   = val if param_name == "t"      else base_t
        results[val] = monte_carlo_simulation(
            lam_use, t_use, severity_fn, n_simulations, seed
        )
    return results


def theoretical_mean(lam: float, t: float, mean_severity: float) -> float:
    """
    E[S(t)] = lambda * t * E[X]

    Wald's identity applied to the compound Poisson sum.
    Example: 10 claims/yr × $1,000 avg → E[S(1)] = $10,000.
    """
    return lam * t * mean_severity


def theoretical_variance(
    lam: float, t: float, mean_severity: float, var_severity: float
) -> float:
    """
    Var[S(t)] = lambda * t * E[X²]
              = lambda * t * (Var[X] + E[X]²)

    The compound Poisson variance formula. Variance is larger than a
    pure Poisson because claim SIZE randomness adds extra spread.
    """
    second_moment = var_severity + mean_severity ** 2
    return lam * t * second_moment


def theoretical_std(
    lam: float, t: float, mean_severity: float, var_severity: float
) -> float:
    """Standard deviation of S(t) = sqrt(Var[S(t)])."""
    return float(np.sqrt(theoretical_variance(lam, t, mean_severity, var_severity)))
