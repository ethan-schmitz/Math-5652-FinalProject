"""
distributions.py

Factory functions that return severity_fn callables of the signature:

    severity_fn(n: int, rng: np.random.Generator) -> np.ndarray of shape (n,)

Each factory creates and returns one such function, with all
its parameters. This design keeps simulation.py clean and makes it easier
to swap or compare severity models.

Supported distributions
1. Exponential – memoryless, simplest model, light tail
2. Lognormal – heavier tail, common for property/casualty losses
3. Gamma – flexible two-parameter model
4. Pareto – heavy-tailed, used for catastrophic / reinsurance losses
5. Weibull – flexible shape, used in reliability and loss modeling

All distributions are parameterised by their MEAN (and optionally a shape /
spread parameter) so that comparisons across distributions are apples-to-apples.
"""

import numpy as np

def exponential_severity(mean: float):
    """
    Exponential severity with the given mean.

    The exponential distribution is memoryless: knowing that a claim exceeds
    some threshold tells you nothing about how much more it will exceed it.
    This is the simplest severity model and a common starting point.

    Mean = mean
    Var = mean²
    """
    def _sample(n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.exponential(scale=mean, size=n)
    _sample.__name__ = f"Exponential(mean={mean})"
    _sample.mean = mean
    _sample.variance = mean ** 2
    return _sample


def lognormal_severity(mean: float, cv: float = 1.0):
    """
    Lognormal severity parameterised by its mean and coefficient of variation.

    CV = std / mean.  cv = 1.0 is a common default in insurance.
    The lognormal produces a right-skewed distribution — most claims are
    small but occasionally very large ones occur, which is realistic.

    Given mean m and cv c:
      sigma² = ln(1 + c²)
      mu     = ln(m) - sigma²/2

    Mean  = exp(mu + sigma²/2) = mean  (by construction)
    Var   = (cv * mean)²
    """
    sigma2 = np.log(1 + cv ** 2)
    sigma = np.sqrt(sigma2)
    mu = np.log(mean) - sigma2 / 2.0

    def _sample(n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.lognormal(mean=mu, sigma=sigma, size=n)

    _sample.__name__ = f"Lognormal(mean={mean}, cv={cv})"
    _sample.mean = mean
    _sample.variance = (cv * mean) ** 2
    return _sample

def gamma_severity(mean: float, cv: float = 1.0):
    """
    Gamma severity parameterised by mean and coefficient of variation.

    When cv=1 this reduces to Exponential. For cv<1 it has a lighter tail;
    for cv>1 it has a heavier one.  The Gamma is very commonly used in
    actuarial loss modelling due to its flexibility.

    shape alpha = 1/cv²
    scale theta = mean * cv²
    """
    alpha = 1.0 / cv ** 2
    theta = mean * cv ** 2

    def _sample(n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.gamma(shape=alpha, scale=theta, size=n)

    _sample.__name__ = f"Gamma(mean={mean}, cv={cv})"
    _sample.mean = mean
    _sample.variance = (cv * mean) ** 2
    return _sample


def pareto_severity(mean: float, alpha: float = 3.0):
    """
    Single-parameter Pareto distribution.

    The Pareto is heavy-tailed, meaning extreme losses are far more likely
    than under exponential or lognormal.  It's used in reinsurance pricing
    and catastrophe modelling.

    For a Pareto with shape alpha and scale xm:
      mean = alpha * xm / (alpha - 1)   [requires alpha > 1]
      var = alpha * xm² / ((alpha-1)²*(alpha-2))  [requires alpha > 2]

    We solve for xm given the desired mean.
    """
    if alpha <= 1:
        raise ValueError("Pareto requires alpha > 1 for finite mean.")
    xm = mean * (alpha - 1) / alpha  # scale such that E[X] = mean

    def _sample(n: int, rng: np.random.Generator) -> np.ndarray:
        # scipy-style Pareto: X = xm * (U^(-1/alpha))  where U ~ Uniform(0,1)
        u = rng.uniform(0, 1, size=n)
        return xm * (u ** (-1.0 / alpha))

    if alpha > 2:
        var = alpha * xm ** 2 / ((alpha - 1) ** 2 * (alpha - 2))
    else:
        var = float("inf")

    _sample.__name__ = f"Pareto(mean={mean}, alpha={alpha})"
    _sample.mean = mean
    _sample.variance = var
    return _sample


def weibull_severity(mean: float, shape: float = 1.5):
    """
    Weibull severity parameterised by mean and shape k.

    shape < 1 → heavy tail; shape = 1 → Exponential; shape > 1 → lighter tail.
    The scale is set so that E[X] matches the desired mean.

    E[X] = scale * Gamma(1 + 1/k)  →  scale = mean / Gamma(1 + 1/k)
    """
    from scipy.special import gamma as gamma_fn
    scale = mean / gamma_fn(1 + 1.0 / shape)

    def _sample(n: int, rng: np.random.Generator) -> np.ndarray:
        return scale * rng.weibull(shape, size=n)

    from scipy.special import gamma as gf
    var = scale ** 2 * (gf(1 + 2.0 / shape) - gf(1 + 1.0 / shape) ** 2)

    _sample.__name__ = f"Weibull(mean={mean}, shape={shape})"
    _sample.mean = mean
    _sample.variance = float(var)
    return _sample



SEVERITY_DISTRIBUTIONS = {
    "exponential": exponential_severity,
    "lognormal":   lognormal_severity,
    "gamma":       gamma_severity,
    "pareto":      pareto_severity,
    "weibull":     weibull_severity,
}
