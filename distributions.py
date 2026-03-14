
"""
Distributions used in liu24z.pdf experiments:
- Uniform(0,1)
- Truncated Normal: X = Y/2 + 1/2 conditioned on |Y| < 1, with Y ~ N(0,1)
- Continuous Bernoulli CB(lambda), lambda=1/4 in the paper

Provides sampler + CDF (+ inverse-CDF when available).
Inverse-CDF lets us reproduce the paper's "G=F" (oracle) sampling baseline (marked with * in appendix).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import numpy as np

try:
    from scipy.stats import norm
except Exception:
    norm = None


Array = np.ndarray


@dataclass(frozen=True)
class Dist:
    name: str
    # Label used in paper-style legends (Figure 1 / 4).
    paper_label: str
    sample: Callable[[int, np.random.Generator], Array]
    cdf: Callable[[Array], Array]
    pdf: Optional[Callable[[Array], Array]] = None
    inv_cdf: Optional[Callable[[Array], Array]] = None  # inverse CDF on u in [0,1]


def _require_scipy():
    if norm is None:
        raise ImportError("scipy is required for the truncated-normal CDF/PPF implementation")


def uniform_dist() -> Dist:
    def sample(n: int, rng: np.random.Generator) -> Array:
        return rng.random(n)

    def cdf(x: Array) -> Array:
        x = np.asarray(x, dtype=float)
        return np.clip(x, 0.0, 1.0)

    def inv_cdf(u: Array) -> Array:
        return np.asarray(u, dtype=float)

    def pdf(x: Array) -> Array:
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x, dtype=float)
        out[(0.0 <= x) & (x <= 1.0)] = 1.0
        return out

    return Dist("U(0,1)", "uniform", sample, cdf, pdf=pdf, inv_cdf=inv_cdf)


def truncated_normal_dist() -> Dist:
    """
    Paper: X = Y/2 + 1/2, conditioned on |Y| < 1, where Y ~ N(0,1).
    """

    _require_scipy()
    Phi_m1 = float(norm.cdf(-1.0))
    Phi_p1 = float(norm.cdf(1.0))
    Z = Phi_p1 - Phi_m1

    def sample(n: int, rng: np.random.Generator) -> Array:
        # Rejection sampling; acceptance prob ~ 0.6827
        out = np.empty(n, dtype=float)
        filled = 0
        while filled < n:
            m = int((n - filled) / 0.68) + 10
            y = rng.standard_normal(m)
            y = y[np.abs(y) < 1.0]
            take = min(len(y), n - filled)
            out[filled:filled + take] = y[:take]
            filled += take
        x = out / 2.0 + 0.5
        return np.clip(x, 0.0, 1.0)

    def cdf(x: Array) -> Array:
        x = np.asarray(x, dtype=float)
        y = 2.0 * x - 1.0
        y = np.clip(y, -1.0, 1.0)
        return (norm.cdf(y) - Phi_m1) / Z

    def inv_cdf(u: Array) -> Array:
        u = np.asarray(u, dtype=float)
        u = np.clip(u, 0.0, 1.0)
        y = norm.ppf(Phi_m1 + u * Z)          # truncated Y in [-1,1]
        x = y / 2.0 + 0.5
        return np.clip(x, 0.0, 1.0)

    def pdf(x: Array) -> Array:
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x, dtype=float)
        mask = (0.0 <= x) & (x <= 1.0)
        y = 2.0 * x[mask] - 1.0
        # X = Y/2 + 1/2, so f_X(x) = f_Y(y) * |dy/dx| = (phi(y)/Z) * 2
        out[mask] = 2.0 * norm.pdf(y) / Z
        return out

    return Dist(
        "Nc(0,1,μ=1/2,σ^2=1/4)",
        "truncated normal",
        sample,
        cdf,
        pdf=pdf,
        inv_cdf=inv_cdf,
    )


def continuous_bernoulli_dist(lam: float = 0.25) -> Dist:
    """
    Continuous Bernoulli CB(λ) on [0,1], λ in (0,1).
    Density: f(x) ∝ λ^x (1-λ)^(1-x) (normalization omitted here).
    Closed-form CDF:
      q = λ/(1-λ);  F(x) = (q^x - 1)/(q - 1), for λ != 0.5.
    Inverse CDF:
      x = log(1 + u*(q-1)) / log(q).
    """

    if not (0.0 < lam < 1.0):
        raise ValueError("lam must be in (0,1)")

    if abs(lam - 0.5) < 1e-12:
        return uniform_dist()

    q = lam / (1.0 - lam)
    a = float(np.log(q))

    def sample(n: int, rng: np.random.Generator) -> Array:
        u = rng.random(n)
        return inv_cdf(u)

    def cdf(x: Array) -> Array:
        x = np.asarray(x, dtype=float)
        x = np.clip(x, 0.0, 1.0)
        return (np.power(q, x) - 1.0) / (q - 1.0)

    def inv_cdf(u: Array) -> Array:
        u = np.asarray(u, dtype=float)
        u = np.clip(u, 0.0, 1.0)
        x = np.log1p(u * (q - 1.0)) / a
        return np.clip(x, 0.0, 1.0)

    def pdf(x: Array) -> Array:
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x, dtype=float)
        mask = (0.0 <= x) & (x <= 1.0)
        # Normalizing constant for density proportional to q^x on [0,1]
        # C = log(q)/(q-1) for q != 1
        C = float(np.log(q) / (q - 1.0))
        out[mask] = C * np.power(q, x[mask])
        return out

    return Dist(f"CB(λ={lam})", "continuous bernoulli", sample, cdf, pdf=pdf, inv_cdf=inv_cdf)


def uniform_G_inv(u: Array) -> Array:
    """Inverse CDF for G(x)=x."""
    return np.asarray(u, dtype=float)


def make_uniform_grid(kappa: int) -> Tuple[Array, Array]:
    """
    Grid and probs for preselected sampling with uniform G.
    Uses interior grid points x_j = j/(kappa+1), j=1..kappa
    (so the grid stays inside (0,1) and avoids the boundary point 1),
    and equal probabilities 1/kappa.
    """
    grid = np.arange(1, kappa + 1, dtype=float) / float(kappa + 1)
    probs = np.ones(kappa, dtype=float) / float(kappa)
    return grid, probs
