# Compound Poisson Insurance Loss Simulation

**Course:** Stochastic Processes — Math 5652  
**Author:** Ethan Schmitz  
**Date:** Spring 2026

---

## Overview

My fianl project is modling insurance losses an actuary like I'm studying to be would see. The losses use a compound poisson proccess, which we have individually seen in class. In my actuary math in practice class we have seen similar models to what I have worked with in this project. One way the insurance companies will calculate their projected ultimate losses is through frequency/severity method which involves using a model like a poisson to find averages to genarlize their patrons costs. Specificaly the following formula was the base framework and idea behind the project like we have seen throughout this semester:

$$S(t) = \sum_{i=1}^{N(t)} X_i$$

- **N(t) ~ Poisson(λt)** — represents a random number of claims arriving in [0, t] (Frequency)
- **X<sub>i</sub> ~ F<sub>X</sub>** — gives random claim sizes, independent of N (Severity)

I used a "Monte Carlo simulation" with 50,000 paths, to estimate what a full distribution of total losses S(t) would look like. Also, I wanted compute key actuarial risk metrics that I may see in the future, and didn't want to rely on a single point deterministic estimate. As you'll see later I didn't want to just look at a single distribution either as actuaries may use different kinds to estimate these claims and numbers. As I have taken the Probability exam for SOA I have learned more about different distributions we didn't necessarily talk about in class, but Iin my research for this project I found other distributions and included them in my final product. There are 5 total distributions in total that I looked at for the simulations. I also looked into plots and what good visualizations would be for some of these numbrers as looking at straight data sometimes doesn't give you the whole picture. So through running the program you will be given some figures that output the final numbers in graphical plots.

**As it is somewhat complicated to run the program if you don't have the right computer or programs I will input a file of outputs I have gotten through running the simulations if there is no time to run the whole program**

---

## The Project's Structure

```
Math-5652-Final-Project/
─ main.py             # THis is the function you run. It executes all the sections and saves and outputs the figures.
─ simulation.py       # Core Monte Carlo simulation runner and includes theoretical moments
─ distributions.py    # Includes the claim severity distributions (Exp, Lognormal, Gamma, Pareto, Weibull)
─ risk_analysis.py    # Includes risk metrics: VaR, CVaR, ruin probability, stop-loss premium
─ visualization.py    # All of the matplotlib that plot the functions
─ requirements.txt    # Python dependencies used throughout
─ figures/            # Output the plots
```

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/ethan-schmitz/Math-5652-FinalProject.git
cd compound-poisson-insurance

# 2. Create a virtual environment (what i recommend)
python -m venv venv
source venv/bin/activate       # for macOS/Linux
venv\Scripts\activate          # for Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the simulation
python main.py
```

The figures are saved to the figures file in your folder and a full risk summary is printed to the console.

---

## Model Details

### 1. Claim Frequency

The number of claims N(t) is modeled as a **Poisson process** with rate λ:

$$N(t) \sim \text{Poisson}(\lambda t)$$

### 2. Claim Severity

Individual claim sizes X<sub>i</sub> are drawn i.i.d. from a chosen distribution. Four are compared:

| Distribution | Tail Behavior* | Actuarial Use |
|---|---|---|
| Exponential | Light (memoryless) | Simple baseline |
| Lognormal | Medium–heavy | Property & casualty |
| Gamma | Flexible | General loss modeling |
| Pareto | Heavy | Catastrophe / reinsurance |

*Tail behavior is just an actuarial term and a number that is used to predict what future claims may look like based on past data

### 3. Theoretical Moments

By Wald's identity for compound Poisson sums:

$$\mathbb{E}[S(t)] = \lambda t \cdot \mathbb{E}[X]$$

$$\text{Var}[S(t)] = \lambda t \cdot \mathbb{E}[X^2] = \lambda t \cdot (\text{Var}[X] + \mathbb{E}[X]^2)$$

The simulation is validated against these expressions.

---

## Risk Metrics

| Metric | Symbol | Description |
|---|---|---|
| Value at Risk | VaR<sub>p</sub> | p-th quantile of S(t) |
| Conditional VaR | CVaR<sub>p</sub> | E[S(t) \| S(t) > VaR<sub>p</sub>] |
| Ruin Probability | ψ(u) | P(S(t) > u) |
| Stop-Loss Premium | π(d) | E[max(S(t) − d, 0)] |
| Safety Loading | — | VaR<sub>p</sub> − E[S(t)] |

---

## The Outputs and Their Use

File | Description |
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

## An Example of What the Console Outputs

```

SECTION 1: Base Compound Poisson Simulation

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


## AI Usage
```
Within this project I used AI for a few helpful structure setups. One of my uses of AI was to generate a format of how a simulation of multiple events and what helper functions I would need in order to code this sort of project. With this it helped me find useful plug ins and imports that can level up what kind of distributions I can use as there are were more intricate functions that aren't just in basic python. As I am a computer science minor I do know a good amount of code and the general idea of how to code, but with this help I was able to achieve something that wouldn't really be in my knowledge range as of now. The other use was to format this README document. I've used github once previously and rembered somehow to set it up and looked up how to import the files and things from visual studio, but I didn't know how to format this document. I used my own research on a helper website and a little bit of AI to make it look better and portray the things I wanted. These were the two things I used AI to assist with my project.
```
---

## References

1. Essentials of Stochastic Processes Third Edition (Richard Durrett)
2. https://matplotlib.org/stable/api/pyplot_summary.html
3. https://www.kwcsangli.in/uploads/4--introduction-to-probability-model-s.ross-math-cs.blog_.ir_.pdf
4. https://mukuba2002.wordpress.com/wp-content/uploads/2012/03/44850471215775.pdf
5. https://ndl.ethernet.edu.et/bitstream/123456789/30397/1/141.Alexander%20J.McNeil.pdf
6. https://www.cns.nyu.edu/csh/csh04/Handouts/Wald_Identity.pdf
7. https://www.investopedia.com/terms/m/montecarlosimulation.asp
8. https://www.freecodecamp.org/news/how-to-write-a-good-readme-file/
9. https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax
