"""
visualization.py
================
All plotting functions for the compound Poisson insurance loss project.

Each function takes a simulated loss array (or sweep dict) as input and
produces a matplotlib figure.  Call plt.show() or fig.savefig() after.

Plots implemented
-----------------
1.  plot_loss_distribution   – histogram + KDE of S(t)
2.  plot_var_cvar             – annotated histogram with VaR/CVaR lines
3.  plot_ecdf                 – empirical CDF with confidence bands
4.  plot_severity_comparison  – overlaid histograms for multiple severity dists.
5.  plot_sensitivity          – boxplots or lines for a parameter sweep
6.  plot_convergence          – how mean/VaR stabilise as n_sim increases
7.  plot_ruin_curve           – ruin probability vs. threshold level
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from typing import Optional

# Consistent colour palette
COLORS = {
    "primary":    "#2563EB",  # blue
    "secondary":  "#16A34A",  # green
    "danger":     "#DC2626",  # red
    "warning":    "#D97706",  # amber
    "muted":      "#6B7280",  # gray
    "light":      "#E5E7EB",
}

plt.rcParams.update({
    "figure.dpi":       120,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "font.size":        11,
})


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _kde(losses: np.ndarray, n_points: int = 500):
    """Return (x_grid, density) for a kernel-density estimate."""
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(losses)
    x   = np.linspace(losses.min(), losses.max(), n_points)
    return x, kde(x)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Loss Distribution  (histogram + KDE)
# ─────────────────────────────────────────────────────────────────────────────

def plot_loss_distribution(
    losses: np.ndarray,
    title: str = "Distribution of Aggregate Insurance Losses S(t)",
    bins: int = 80,
    show_normal_approx: bool = True,
) -> plt.Figure:
    """
    Plot histogram of simulated aggregate losses with an overlaid KDE.

    Optionally adds a Normal approximation (same mean & std) to illustrate
    why the Normal is a poor model for insurance losses (right skew, heavy tail).
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # --- Histogram ---
    ax.hist(losses, bins=bins, density=True,
            color=COLORS["primary"], alpha=0.4, label="Simulated losses")

    # --- KDE overlay ---
    x_kde, y_kde = _kde(losses)
    ax.plot(x_kde, y_kde, color=COLORS["primary"], lw=2, label="KDE")

    # --- Normal approximation ---
    if show_normal_approx:
        from scipy.stats import norm
        mu, sigma = np.mean(losses), np.std(losses)
        x_norm = np.linspace(losses.min(), losses.max(), 400)
        ax.plot(x_norm, norm.pdf(x_norm, mu, sigma),
                color=COLORS["danger"], lw=1.5, ls="--",
                label=f"Normal approx  (μ={mu:,.0f}, σ={sigma:,.0f})")

    # --- Mean line ---
    ax.axvline(np.mean(losses), color=COLORS["muted"], ls=":", lw=1.5,
               label=f"Mean = ${np.mean(losses):,.2f}")

    ax.set_xlabel("Aggregate Loss  S(t)  ($)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. VaR / CVaR Annotated Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_var_cvar(
    losses: np.ndarray,
    confidence: float = 0.95,
    bins: int = 80,
) -> plt.Figure:
    """
    Histogram with VaR and CVaR annotated, and the tail region shaded.

    This is the most important visualisation for communicating risk to
    non-technical stakeholders (regulators, senior management).
    """
    from risk_analysis import value_at_risk, conditional_var

    var  = value_at_risk(losses, confidence)
    cvar = conditional_var(losses, confidence)

    fig, ax = plt.subplots(figsize=(10, 5))

    # --- Base histogram ---
    n, bins_arr, patches = ax.hist(losses, bins=bins, density=True,
                                   color=COLORS["light"], edgecolor="white")

    # --- Shade the tail (losses > VaR) ---
    for patch, left in zip(patches, bins_arr[:-1]):
        if left >= var:
            patch.set_facecolor(COLORS["danger"])
            patch.set_alpha(0.7)

    # --- KDE ---
    x_kde, y_kde = _kde(losses)
    ax.plot(x_kde, y_kde, color=COLORS["primary"], lw=2)

    # --- VaR line ---
    ax.axvline(var, color=COLORS["warning"], lw=2, ls="--",
               label=f"VaR {int(confidence*100)}% = ${var:,.2f}")

    # --- CVaR line ---
    ax.axvline(cvar, color=COLORS["danger"], lw=2, ls="-",
               label=f"CVaR {int(confidence*100)}% = ${cvar:,.2f}")

    # --- Shade label ---
    tail_prob = 1 - confidence
    ax.annotate(
        f"Tail ({tail_prob*100:.0f}% of scenarios)",
        xy=(var + (cvar - var) * 0.5, ax.get_ylim()[1] * 0.6),
        fontsize=9, color=COLORS["danger"], ha="center",
    )

    ax.set_xlabel("Aggregate Loss  S(t)  ($)")
    ax.set_ylabel("Density")
    ax.set_title(f"Aggregate Loss Distribution with {int(confidence*100)}% VaR and CVaR")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. Empirical CDF
# ─────────────────────────────────────────────────────────────────────────────

def plot_ecdf(
    losses: np.ndarray,
    confidence_levels: Optional[list] = None,
) -> plt.Figure:
    """
    Empirical CDF of aggregate losses with horizontal lines at common
    confidence levels.  Useful for reading off VaR at any level visually.
    """
    if confidence_levels is None:
        confidence_levels = [0.90, 0.95, 0.99]

    sorted_losses = np.sort(losses)
    ecdf_values   = np.arange(1, len(losses) + 1) / len(losses)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sorted_losses, ecdf_values,
            color=COLORS["primary"], lw=1.5, label="Empirical CDF")

    for p in confidence_levels:
        var = float(np.quantile(losses, p))
        ax.axhline(p, color=COLORS["muted"], ls=":", lw=1)
        ax.axvline(var, color=COLORS["muted"], ls=":", lw=1)
        ax.annotate(f"VaR {int(p*100)}% = ${var:,.0f}",
                    xy=(var, p), xytext=(var * 1.01, p - 0.03),
                    fontsize=8, color=COLORS["muted"])

    ax.set_xlabel("Aggregate Loss  S(t)  ($)")
    ax.set_ylabel("Cumulative Probability  F(x)")
    ax.set_title("Empirical CDF of Aggregate Losses")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. Severity Distribution Comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_severity_comparison(
    loss_arrays: dict,   # { label: np.ndarray }
    bins: int = 80,
) -> plt.Figure:
    """
    Overlay KDEs for multiple simulated loss distributions on one plot.
    Useful for showing how the choice of severity distribution impacts risk.

    Parameters
    ----------
    loss_arrays : dict mapping distribution name → simulated loss array.
    """
    palette = [COLORS["primary"], COLORS["secondary"],
               COLORS["danger"], COLORS["warning"], COLORS["muted"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- KDE overlays ---
    for (label, losses), color in zip(loss_arrays.items(), palette):
        x_kde, y_kde = _kde(losses)
        axes[0].plot(x_kde, y_kde, lw=2, color=color, label=label)
        axes[0].axvline(np.mean(losses), ls=":", lw=1, color=color)

    axes[0].set_xlabel("Aggregate Loss  S(t)  ($)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Loss Distributions by Severity Model")
    axes[0].legend(fontsize=9)

    # --- VaR comparison bar chart ---
    labels  = list(loss_arrays.keys())
    var_99  = [float(np.quantile(v, 0.99)) for v in loss_arrays.values()]
    cvar_99 = []
    for v in loss_arrays.values():
        threshold = float(np.quantile(v, 0.99))
        tail = v[v > threshold]
        cvar_99.append(float(np.mean(tail)) if len(tail) > 0 else threshold)

    x = np.arange(len(labels))
    w = 0.35
    axes[1].bar(x - w/2, var_99,  width=w, label="VaR 99%",  color=COLORS["warning"], alpha=0.8)
    axes[1].bar(x + w/2, cvar_99, width=w, label="CVaR 99%", color=COLORS["danger"],  alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15, ha="right")
    axes[1].set_ylabel("Loss ($)")
    axes[1].set_title("VaR and CVaR at 99% by Severity Model")
    axes[1].legend(fontsize=9)

    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5. Sensitivity Analysis
# ─────────────────────────────────────────────────────────────────────────────

def plot_sensitivity(
    sweep_results: dict,  # { param_value : np.ndarray }
    param_name: str,
    confidence: float = 0.95,
) -> plt.Figure:
    """
    Plot how mean, VaR, and CVaR change as a parameter is swept.

    This is sensitivity analysis — identifying which parameters most
    strongly drive risk.
    """
    from risk_analysis import value_at_risk, conditional_var

    param_vals = sorted(sweep_results.keys())
    means  = [np.mean(sweep_results[v])               for v in param_vals]
    vars_  = [value_at_risk(sweep_results[v], confidence)  for v in param_vals]
    cvars  = [conditional_var(sweep_results[v], confidence) for v in param_vals]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(param_vals, means,  "o-", color=COLORS["primary"],   lw=2, label="Mean E[S(t)]")
    ax.plot(param_vals, vars_,  "s--", color=COLORS["warning"],  lw=2, label=f"VaR {int(confidence*100)}%")
    ax.plot(param_vals, cvars,  "^-.", color=COLORS["danger"],   lw=2, label=f"CVaR {int(confidence*100)}%")

    ax.set_xlabel(param_name)
    ax.set_ylabel("Loss ($)")
    ax.set_title(f"Sensitivity Analysis: Risk Metrics vs. {param_name}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6. Convergence of Monte Carlo Estimates
# ─────────────────────────────────────────────────────────────────────────────

def plot_convergence(losses: np.ndarray, confidence: float = 0.95) -> plt.Figure:
    """
    Show how the running mean and running VaR converge as more simulations
    are added.  Validates that 50,000 simulations is sufficient.
    """
    n = len(losses)
    steps = np.arange(100, n + 1, max(1, n // 200))

    running_mean = [np.mean(losses[:k]) for k in steps]
    running_var  = [float(np.quantile(losses[:k], confidence)) for k in steps]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    ax1.plot(steps, running_mean, color=COLORS["primary"], lw=1.5)
    ax1.axhline(np.mean(losses), color=COLORS["muted"], ls="--", lw=1,
                label=f"Final mean = ${np.mean(losses):,.2f}")
    ax1.set_xlabel("Number of simulations")
    ax1.set_ylabel("Running mean E[S(t)] ($)")
    ax1.set_title("Convergence of Mean Estimate")
    ax1.legend(fontsize=9)

    ax2.plot(steps, running_var, color=COLORS["warning"], lw=1.5)
    ax2.axhline(float(np.quantile(losses, confidence)),
                color=COLORS["muted"], ls="--", lw=1,
                label=f"Final VaR = ${np.quantile(losses, confidence):,.2f}")
    ax2.set_xlabel("Number of simulations")
    ax2.set_ylabel(f"Running VaR {int(confidence*100)}% ($)")
    ax2.set_title(f"Convergence of VaR {int(confidence*100)}% Estimate")
    ax2.legend(fontsize=9)

    fig.suptitle("Monte Carlo Convergence Diagnostics", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 7. Ruin Probability Curve
# ─────────────────────────────────────────────────────────────────────────────

def plot_ruin_curve(losses: np.ndarray, n_thresholds: int = 200) -> plt.Figure:
    """
    Plot P(S(t) > x) as a function of x -- the survival / ruin function.

    This is the key output actuaries use to set reserve levels: 'what reserve
    do we need so that the probability of insolvency is below 0.5%?'
    """
    max_loss = np.percentile(losses, 99.5)
    thresholds = np.linspace(0, max_loss, n_thresholds)
    ruin_probs = [float(np.mean(losses > thr)) for thr in thresholds]

    # Find reserve for 1% and 0.5% ruin probability
    target_probs = {0.01: None, 0.005: None}
    for thr, rp in zip(thresholds, ruin_probs):
        for target in target_probs:
            if target_probs[target] is None and rp <= target:
                target_probs[target] = thr

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thresholds, ruin_probs, color=COLORS["primary"], lw=2, label="P(S(t) > x)")
    ax.fill_between(thresholds, ruin_probs, alpha=0.1, color=COLORS["primary"])

    for target, reserve in target_probs.items():
        if reserve is not None:
            ax.axvline(reserve, color=COLORS["danger"], ls="--", lw=1.5,
                       label=f"{target*100:.1f}% ruin level: ${reserve:,.0f}")
            ax.axhline(target, color=COLORS["danger"], ls=":", lw=1)

    ax.set_xlabel("Reserve Level / Loss Threshold  ($)")
    ax.set_ylabel("Ruin Probability  P(S(t) > x)")
    ax.set_title("Ruin Probability Curve")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig