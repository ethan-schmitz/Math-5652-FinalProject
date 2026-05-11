"""
main.py

Entry point for the Compound Poisson Insurance Loss Simulation project.

What this script does
1. Defines base model parameters (lambda, t, severity distribution).
2. Runs the Monte Carlo simulation (50,000 paths).
3. Computes and prints all key risk metrics.
4. Validates the simulation against closed-form theoretical moments.
5. Compares four severity distributions (Exp, Lognormal, Gamma, Pareto).
6. Runs sensitivity analysis on the arrival rate lambda.
7. Saves all figures to ./figures/.

How to run
    python main.py

Output
  - Console: printed summary tables
  - ./figures/: all plots as .png files
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from simulation    import (monte_carlo_simulation, parameter_sweep,
                           theoretical_mean, theoretical_variance,
                           theoretical_std)
from distributions import (exponential_severity, lognormal_severity,
                           gamma_severity, pareto_severity)
from risk_analysis import (descriptive_stats, value_at_risk, conditional_var,
                           ruin_probability, stop_loss_premium,
                           safety_loading, full_summary)
from visualization import (plot_loss_distribution, plot_var_cvar,
                           plot_ecdf, plot_severity_comparison,
                           plot_sensitivity, plot_convergence,
                           plot_ruin_curve)


# Base model parameters
LAM  = 10    # arrival rate: 10 claims per year on average
T    = 1.0    # observation window: 1 year
MEAN_SEVERITY = 1_000.0    # average claim size: $1,000
CV_SEVERITY   = 1.5    # coefficient of variation of claim sizes
N_SIM = 50_000    # Monte Carlo replications
SEED  = 42    # for full reproducibility

# Output directory for figures
FIG_DIR = "./figures"
os.makedirs(FIG_DIR, exist_ok=True)


def save(fig: plt.Figure, name: str):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    print(f"  [saved] {path}")
    plt.close(fig)


def section_1_base_simulation():
    print("\n" + "="*65)
    print("SECTION 1: Base Compound Poisson Simulation")
    print("="*65)
    print(f"  lambda = {LAM} claims/yr | t = {T} yr | "
          f"Mean severity = ${MEAN_SEVERITY:,.0f} | CV = {CV_SEVERITY}")

    # Build severity function
    sev_fn = lognormal_severity(mean=MEAN_SEVERITY, cv=CV_SEVERITY)

    # Run Monte Carlo
    print(f"\n  Running {N_SIM:,} Monte Carlo simulations ...")
    losses = monte_carlo_simulation(LAM, T, sev_fn, N_SIM, SEED)

    # Simulated vs theoretical moments 
    theo_mean = theoretical_mean(LAM, T, sev_fn.mean)
    theo_std  = theoretical_std(LAM, T, sev_fn.mean, sev_fn.variance)

    print("\n  Simulated vs. Theoretical Moments:")
    print(f"    Mean:   simulated = ${np.mean(losses):>10,.2f}  |  "
          f"theoretical = ${theo_mean:>10,.2f}")
    print(f"    Std:    simulated = ${np.std(losses):>10,.2f}  |  "
          f"theoretical = ${theo_std:>10,.2f}")

    # Risk metrics
    print("\n  Key Risk Metrics:")
    for p in [0.90, 0.95, 0.99, 0.995]:
        var  = value_at_risk(losses, p)
        cvar = conditional_var(losses, p)
        print(f"    VaR {int(p*100):3d}% = ${var:>10,.2f}  |  "
              f"CVaR {int(p*100):3d}% = ${cvar:>10,.2f}")

    # Ruin probabilities at various reserve levels
    print("\n  Ruin Probabilities:")
    reserves = [theo_mean * m for m in [1.5, 2.0, 2.5, 3.0]]
    for r in reserves:
        pr = ruin_probability(losses, r)
        print(f"    P(S > ${r:>9,.0f}) = {pr*100:.3f}%")

    # Stop-loss premiums 
    print("\n  Stop-Loss (Reinsurance) Premiums:")
    deductibles = [theo_mean * m for m in [1.0, 1.5, 2.0]]
    for d in deductibles:
        sl = stop_loss_premium(losses, d)
        print(f"    E[max(S - ${d:>8,.0f}, 0)] = ${sl:>8,.2f}")

    # Full summary table
    print("\n  Full Summary Table:")
    df = full_summary(losses, LAM, T, sev_fn)
    print(df.to_string(index=False))

    # Plots 
    print("\n  Generating plots ...")
    save(plot_loss_distribution(losses, title="Aggregate Losses — Lognormal Severity"),
         "01_loss_distribution.png")
    save(plot_var_cvar(losses, confidence=0.95), "02_var_cvar_95.png")
    save(plot_var_cvar(losses, confidence=0.99), "03_var_cvar_99.png")
    save(plot_ecdf(losses), "04_ecdf.png")
    save(plot_convergence(losses), "05_convergence.png")
    save(plot_ruin_curve(losses), "06_ruin_curve.png")

    return losses, sev_fn




def section_2_severity_comparison():
    print("\n" + "="*65)
    print("SECTION 2: Severity Distribution Comparison")
    print("="*65)

    severity_functions = {
        f"Exponential":  exponential_severity(mean=MEAN_SEVERITY),
        f"Lognormal (cv={CV_SEVERITY})": lognormal_severity(mean=MEAN_SEVERITY, cv=CV_SEVERITY),
        f"Gamma (cv={CV_SEVERITY})": gamma_severity(mean=MEAN_SEVERITY, cv=CV_SEVERITY),
        f"Pareto (alpha=3)": pareto_severity(mean=MEAN_SEVERITY, alpha=3.0),
    }

    print(f"\n  Running {N_SIM:,} simulations per distribution ...")
    loss_arrays = {}
    for label, sev_fn in severity_functions.items():
        loss_arrays[label] = monte_carlo_simulation(LAM, T, sev_fn, N_SIM, SEED)
        var99 = value_at_risk(loss_arrays[label], 0.99)
        cvar99 = conditional_var(loss_arrays[label], 0.99)
        print(f"    {label:<30}  VaR99%=${var99:>9,.0f}  CVaR99%=${cvar99:>9,.0f}")

    save(plot_severity_comparison(loss_arrays), "07_severity_comparison.png")


def section_3_sensitivity():
    print("\n" + "="*65)
    print("SECTION 3: Sensitivity Analysis — Varying Claim Frequency λ")
    print("="*65)

    sev_fn = lognormal_severity(mean=MEAN_SEVERITY, cv=CV_SEVERITY)
    lam_values = [2, 5, 10, 20, 50]

    print(f"\n  Sweeping lambda over {lam_values} ...")
    sweep = parameter_sweep(
        base_lam=LAM, base_t=T, severity_fn=sev_fn,
        param_name="lambda", param_values=lam_values,
        n_simulations=20_000, seed=SEED,
    )

    print(f"\n  {'lambda':>8}  {'Mean':>10}  {'VaR 95%':>12}  {'CVaR 95%':>12}")
    print("  " + "-"*48)
    for lam_val in lam_values:
        arr  = sweep[lam_val]
        mean = np.mean(arr)
        var  = value_at_risk(arr, 0.95)
        cvar = conditional_var(arr, 0.95)
        print(f"  {lam_val:>8}  ${mean:>9,.2f}  ${var:>11,.2f}  ${cvar:>11,.2f}")

    save(plot_sensitivity(sweep, param_name="Arrival Rate  λ  (claims/yr)"),
         "08_sensitivity_lambda.png")

    # Also sweep t (time horizon)
    print("\n  Sweeping time horizon t over [0.25, 0.5, 1, 2, 5] ...")
    t_sweep = parameter_sweep(
        base_lam=LAM, base_t=T, severity_fn=sev_fn,
        param_name="t", param_values=[0.25, 0.5, 1.0, 2.0, 5.0],
        n_simulations=20_000, seed=SEED,
    )
    save(plot_sensitivity(t_sweep, param_name="Time Horizon  t  (years)"),
         "09_sensitivity_t.png")


if __name__ == "__main__":
    print("\n" + "#"*65)
    print("  Compound Poisson Insurance Loss Simulation")
    print("  Ethan Schmitz — Stochastic Processes Final Project")
    print("#"*65)

    losses, sev_fn = section_1_base_simulation()
    section_2_severity_comparison()
    section_3_sensitivity()

    print("\n" + "="*65)
    print("All done! Figures saved to ./figures/")
    print("="*65)
