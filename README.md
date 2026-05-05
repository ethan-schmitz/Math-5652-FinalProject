# Compound Poisson Insurance Loss Simulation

**Course:** Stochastic Processes — Final Project  
**Author:** Ethan Schmitz  
**Date:** Spring 2026

---

## Overview

This project models aggregate insurance losses using a **Compound Poisson Process** — the standard actuarial frequency-severity framework:

$$S(t) = \sum_{i=1}^{N(t)} X_i$$

- **N(t) ~ Poisson(λt)** — random number of claims arriving in [0, t]
- **X_i ~ F_X** — i.i.d. random claim sizes, independent of N

Using Monte Carlo simulation (50,000 paths), we estimate the full **distribution** of total losses S(t) and compute key actuarial risk metrics, rather than relying on a single-point deterministic estimate.

---

## Project Structure

```
compound-poisson-insurance/
├── main.py             # ← Run this. Executes all sections and saves figures.
├── simulation.py       # Core Monte Carlo engine + theoretical moments
├── distributions.py    # Claim severity distributions (Exp, Lognormal, Gamma, Pareto, Weibull)
├── risk_analysis.py    # Risk metrics: VaR, CVaR, ruin probability, stop-loss premium
├── visualization.py    # All matplotlib plotting functions
├── requirements.txt    # Python dependencies
└── figures/            # Output plots (auto-created on first run)
```

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/compound-poisson-insurance.git
cd compound-poisson-insurance

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the simulation
python main.py
```

Figures are saved to `./figures/` and a full risk summary is printed to the console.

---

## Model Details

### 1. Claim Frequency

The number of claims N(t) is modeled as a **Poisson process** with rate λ:

$$N(t) \sim \text{Poisson}(\lambda t)$$

### 2. Claim Severity

Individual claim sizes X_i are drawn i.i.d. from a chosen distribution. Four are compared:

| Distribution | Tail Behaviour | Typical Use Case |
|---|---|---|
| Exponential | Light (memoryless) | Simple baseline |
| Lognormal | Medium–heavy | Property & casualty |
| Gamma | Flexible | General loss modeling |
| Pareto | Heavy | Catastrophe / reinsurance |

### 3. Theoretical Moments

By Wald's identity for compound Poisson sums:

$$\mathbb{E}[S(t)] = \lambda t \cdot \mathbb{E}[X]$$

$$\text{Var}[S(t)] = \lambda t \cdot \mathbb{E}[X^2] = \lambda t \cdot (\text{Var}[X] + \mathbb{E}[X]^2)$$

The simulation is validated against these closed-form expressions.

---

## Risk Metrics

| Metric | Symbol | Description |
|---|---|---|
| Value at Risk | VaR_p | p-th quantile of S(t) |
| Conditional VaR | CVaR_p | E[S(t) \| S(t) > VaR_p] |
| Ruin Probability | ψ(u) | P(S(t) > u) |
| Stop-Loss Premium | π(d) | E[max(S(t) − d, 0)] |
| Safety Loading | — | VaR_p − E[S(t)] |

---

## Output Figures

| File | Description |
|---|---|
| `01_loss_distribution.png` | Histogram + KDE of S(t) with Normal approximation |
| `02_var_cvar_95.png` | Loss distribution with 95% VaR and CVaR annotated |
| `03_var_cvar_99.png` | Loss distribution with 99% VaR and CVaR annotated |
| `04_ecdf.png` | Empirical CDF with VaR levels marked |
| `05_convergence.png` | Monte Carlo convergence of mean and VaR estimates |
| `06_ruin_curve.png` | Ruin probability P(S > x) as a function of reserve x |
| `07_severity_comparison.png` | KDE + VaR/CVaR bar chart for 4 severity distributions |
| `08_sensitivity_lambda.png` | Risk metrics vs. claim arrival rate λ |
| `09_sensitivity_t.png` | Risk metrics vs. time horizon t |

---

## Example Console Output

```
=================================================================
SECTION 1: Base Compound Poisson Simulation
=================================================================
  lambda = 10 claims/yr | t = 1 yr | Mean severity = $1,000 | CV = 1.5

  Running 50,000 Monte Carlo simulations ...

  Simulated vs. Theoretical Moments:
    Mean:   simulated =  $9,987.43  |  theoretical =  $10,000.00
    Std:    simulated =  $6,245.11  |  theoretical =   $6,244.99

  Key Risk Metrics:
    VaR  90% =  $17,432.10  |  CVaR  90% =  $23,891.55
    VaR  95% =  $21,042.30  |  CVaR  95% =  $28,744.20
    VaR  99% =  $29,871.44  |  CVaR  99% =  $38,521.67
    VaR 100% =  $36,881.22  |  CVaR 100% =  $47,200.11
```

---

## Dependencies

```
numpy>=1.24
scipy>=1.10
matplotlib>=3.7
pandas>=2.0
```

---

## References

1. Ross, S.M. *Introduction to Probability Models* — compound Poisson process (Ch. 5)
2. Klugman, S.A. et al. *Loss Models: From Data to Decisions* — frequency-severity framework
3. McNeil, A.J. et al. *Quantitative Risk Management* — VaR, CVaR, and coherent risk measures