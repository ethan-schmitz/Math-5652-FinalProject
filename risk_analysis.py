"""
risk_analysis.py

Actuarial risk metrics computed from an empirical loss distribution
(a numpy array produced by simulation.py).

Metrics implemented
1.  Descriptive summary – mean, std, skewness, kurtosis
2.  Value at Risk (VaR) – loss threshold not exceeded with probability p
3.  Conditional VaR (CVaR) – expected loss GIVEN we are in the tail (TVaR)
4.  Probability of ruin – P(S(t) > threshold)
5.  Stop-loss premium – E[max(S(t) - d, 0)]  (excess-of-loss reinsurance)
6.  Safety loading – how much capital above the mean is needed
7.  Full summary table – DataFrame with all key stats
"""

import numpy as np
import pandas as pd
from typing import Optional


def descriptive_stats(losses: np.ndarray) -> dict:
    """
    Return a dict of basic descriptive statistics for a loss array.

    Skewness > 0 (right-skewed) is expected for insurance losses because
    there is no upper bound but there is a lower bound of 0.
    """
    from scipy import stats as sp_stats
    return {
        "n_simulations": len(losses),
        "mean": float(np.mean(losses)),
        "std": float(np.std(losses)),
        "median": float(np.median(losses)),
        "min": float(np.min(losses)),
        "max": float(np.max(losses)),
        "skewness": float(sp_stats.skew(losses)),
        "kurtosis": float(sp_stats.kurtosis(losses)),  # excess kurtosis
        "p_zero_loss": float(np.mean(losses == 0.0)),     # P(N=0)
    }


def value_at_risk(losses: np.ndarray, confidence: float = 0.95) -> float:
    """
    VaR_p = smallest loss level L such that P(S(t) <= L) >= p.

    Equivalently, the p-th quantile of the empirical loss distribution.

    In insurance and banking this is the standard regulatory risk measure
    (Basel III uses 99% VaR; Solvency II uses 99.5% VaR).

    Parameters
    losses: Simulated aggregate loss array.
    confidence: Probability level p (e.g. 0.95, 0.99, 0.995).

    Returns
    float: VaR at the given confidence level.
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1).")
    return float(np.quantile(losses, confidence))

def conditional_var(losses: np.ndarray, confidence: float = 0.95) -> float:
    """
    CVaR_p = E[S(t) | S(t) > VaR_p]

    Also known as Tail Value at Risk (TVaR) or Expected Shortfall (ES).
    CVaR is a COHERENT risk measure (unlike VaR), meaning it properly
    accounts for the shape of the tail beyond the VaR threshold.

    Actuaries prefer CVaR for capital adequacy because it tells you the
    AVERAGE loss in the worst (1-p) scenarios, not just the threshold.

    Parameters
    losses: Simulated aggregate loss array.
    confidence: Same p as used for VaR.

    Returns
    float: CVaR (expected shortfall) at confidence level p.
    """
    var_threshold = value_at_risk(losses, confidence)
    tail_losses   = losses[losses > var_threshold]

    if len(tail_losses) == 0:
        return var_threshold  # all losses are at or below VaR

    return float(np.mean(tail_losses))
    

def ruin_probability(losses: np.ndarray, threshold: float) -> float:
    """
    P(S(t) > threshold) -- empirical exceedance probability.

    In the classical ruin theory context, this is the probability that the
    insurer's total claims exceed their reserves / premium income.

    Parameters
    losses: Simulated aggregate loss array.
    threshold: The reserve / retention level to test.

    Returns
    float: Estimated probability of ruin.
    """
    return float(np.mean(losses > threshold))



def stop_loss_premium(losses: np.ndarray, deductible: float) -> float:
    """
    E[max(S(t) - d, 0)]

    The stop-loss premium is the fair price of reinsurance that pays the
    insurer for aggregate losses exceeding a deductible d.

    This is a fundamental quantity in excess-of-loss reinsurance pricing.

    Parameters
    losse: Simulated aggregate loss array.
    deductible: The attachment point d.

    Returns
    float: Expected excess loss above the deductible.
    """
    excess = np.maximum(losses - deductible, 0.0)
    return float(np.mean(excess))



def safety_loading(losses: np.ndarray, confidence: float = 0.99) -> dict:
    """
    Compute the capital buffer needed above the mean to cover losses at
    a given confidence level.

    This is directly relevant to Solvency Capital Requirement (SCR) calculations.

    Returns
    dict with:
      'mean': E[S(t)]
      'var': VaR_p
      'cvar': CVaR_p
      'buffer_var': VaR_p - E[S(t)]  (capital above mean for VaR coverage)
      'buffer_cvar': CVaR_p - E[S(t)] (capital above mean for CVaR coverage)
      'loading_pct': buffer_var / mean * 100  (percentage loading)
    """
    mu   = float(np.mean(losses))
    var  = value_at_risk(losses, confidence)
    cvar = conditional_var(losses, confidence)

    return {
        "confidence": confidence,
        "mean": mu,
        "var": var,
        "cvar": cvar,
        "buffer_var": var  - mu,
        "buffer_cvar": cvar - mu,
        "loading_pct": (var - mu) / mu * 100 if mu > 0 else float("nan"),
    }

def full_summary(
    losses: np.ndarray,
    lam: float,
    t: float,
    severity_fn,
    thresholds: Optional[list] = None,
    confidence_levels: Optional[list] = None,
) -> pd.DataFrame:
    """
    Parameters
    losses: Simulated loss array.
    lam, t: Model parameters (for theoretical benchmarks).
    severity_fn: Severity function with .mean and .variance attributes.
    thresholds: Loss levels for ruin probability computation.
    confidence_levels: Probability levels for VaR/CVaR.

    Returns
    pd.DataFrame with columns ['Metric', 'Simulated', 'Theoretical']
    """
    from simulation import theoretical_mean, theoretical_variance

    if thresholds is None:
        thresholds = [
            np.percentile(losses, 90),
            np.percentile(losses, 95),
            np.percentile(losses, 99),
        ]
    if confidence_levels is None:
        confidence_levels = [0.90, 0.95, 0.99, 0.995]

    theo_mean = theoretical_mean(lam, t, severity_fn.mean)
    theo_std  = float(np.sqrt(
        theoretical_variance(lam, t, severity_fn.mean, severity_fn.variance)
    ))

    rows = []

    # Descriptive
    rows.append(("Mean E[S(t)]",      f"${np.mean(losses):,.2f}",  f"${theo_mean:,.2f}"))
    rows.append(("Std Dev",           f"${np.std(losses):,.2f}",   f"${theo_std:,.2f}"))
    rows.append(("Median",            f"${np.median(losses):,.2f}", "N/A"))
    rows.append(("Zero-loss prob",    f"{np.mean(losses == 0)*100:.2f}%", "N/A"))

    # VaR and CVaR
    for p in confidence_levels:
        var  = value_at_risk(losses, p)
        cvar = conditional_var(losses, p)
        rows.append((f"VaR  {int(p*100)}%",  f"${var:,.2f}",  "—"))
        rows.append((f"CVaR {int(p*100)}%",  f"${cvar:,.2f}", "—"))

    # Ruin probabilities
    for thr in thresholds:
        p_ruin = ruin_probability(losses, thr)
        rows.append((f"P(S > ${thr:,.0f})", f"{p_ruin*100:.3f}%", "—"))

    df = pd.DataFrame(rows, columns=["Metric", "Simulated", "Theoretical"])
    return df
