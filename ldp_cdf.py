
"""
Tuning-free Estimation and Inference of CDF under Local Differential Privacy
Reproduction code for liu24z.pdf (ICML 2024).

Implements:
- LDP data collection via randomized response on binary query 1{X <= T}
- Constrained isotonic estimator (Algorithm 1): GCM/PAVA + linear inversion + clipping
- Experiment utilities for density-based sampling and preselected sampling

Notes:
- The paper's Algorithm 1 (lines 22-48 on p.6) computes the Greatest Convex Minorant (GCM)
  of the cusum diagram (H1, H2) and takes its left-derivative to get \\hat F_b^*(x).
  This is equivalent to the Pool-Adjacent-Violators Algorithm (PAVA) on the grouped proportions.
- Interpolation in experiments uses "nearest T to the left" (left-continuous staircase).

This file aims to be deterministic given a NumPy RNG seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np

Array = np.ndarray


def epsilon_from_r(r: float) -> float:
    """Convert truthful response rate r in (0,1) to epsilon."""
    if not (0 < r < 1):
        raise ValueError("r must be in (0,1)")
    return float(np.log((1 + r) / (1 - r)))


def r_from_epsilon(eps: float) -> float:
    """Convert epsilon > 0 to r = tanh(eps/2)."""
    if eps <= 0:
        raise ValueError("eps must be > 0")
    return float(np.tanh(eps / 2.0))


def ldp_random_response_indicator(x: Array, t: Array, r: float, rng: np.random.Generator) -> Array:
    """
    Definition 3.1 randomized response:
      Δ = 1{x <= t} with prob r,
          Bernoulli(0.5) with prob 1-r.
    """
    if x.shape != t.shape:
        raise ValueError("x and t must have the same shape")
    if not (0 <= r <= 1):
        raise ValueError("r must be in [0,1]")
    truthful = (rng.random(size=x.shape) < r)
    delta = np.empty_like(x, dtype=np.int8)
    # truthful answers
    delta[truthful] = (x[truthful] <= t[truthful]).astype(np.int8)
    # random answers
    delta[~truthful] = (rng.random(size=np.sum(~truthful)) < 0.5).astype(np.int8)
    return delta


@dataclass
class IsotonicCDF:
    """Left-continuous staircase CDF defined on unique T points."""
    t_unique: Array               # strictly increasing
    fhat_at_t: Array              # same length, values in [0,1]

    def eval_left(self, x: Array) -> Array:
        """
        Evaluate using nearest T to the left (left-continuous staircase).
        For x < min(T): return 0.
        For x >= max(T): return fhat_at_t[-1] (NOT forced to 1, matching the paper's 'left' fill).
        """
        x = np.asarray(x)
        idx = np.searchsorted(self.t_unique, x, side="right") - 1
        out = np.zeros_like(x, dtype=float)
        mask = idx >= 0
        out[mask] = self.fhat_at_t[idx[mask]]
        return out


def _pava_non_decreasing(y: Array, w: Array) -> Array:
    """
    Weighted PAVA for non-decreasing sequence.
    Returns fitted values (same length as y).
    """
    # NOTE: This implementation is O(n) using a stack of blocks.
    # The previous array-shifting version can degrade to O(n^2) and becomes
    # impractical for n=1e5.

    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    n = len(y)

    # block starts, weights, and averages
    starts: List[int] = []
    weights: List[float] = []
    avgs: List[float] = []

    for i in range(n):
        starts.append(i)
        weights.append(float(w[i]))
        avgs.append(float(y[i]))

        while len(avgs) >= 2 and avgs[-2] > avgs[-1] + 1e-15:
            w_new = weights[-2] + weights[-1]
            a_new = (weights[-2] * avgs[-2] + weights[-1] * avgs[-1]) / w_new
            weights[-2] = w_new
            avgs[-2] = a_new
            # pop last block
            starts.pop()
            weights.pop()
            avgs.pop()

    fitted = np.empty(n, dtype=float)
    for bi in range(len(avgs)):
        s = starts[bi]
        e = starts[bi + 1] if bi + 1 < len(avgs) else n
        fitted[s:e] = avgs[bi]
    return fitted


def constrained_isotonic_cdf(delta: Array, t: Array, r: float) -> IsotonicCDF:
    """
    Algorithm 1 (constrained isotonic estimation).
    Steps:
      - Group by unique t (handles discrete/preselected sampling).
      - Compute unconstrained isotonic estimate of F_b^*(t) via PAVA on group proportions.
      - Invert: F_e = (F_b^* - (1-r)/2)/r
      - Clip to [0,1].
    Returns a left-continuous staircase evaluator (values only stored at unique t).
    """
    delta = np.asarray(delta).astype(np.int8)
    t = np.asarray(t).astype(float)
    if delta.shape != t.shape:
        raise ValueError("delta and t must have same shape")
    if not (0 < r <= 1):
        raise ValueError("r must be in (0,1] for inversion")

    # sort by t
    order = np.argsort(t, kind="mergesort")
    t_sorted = t[order]
    d_sorted = delta[order]

    # group identical t (important for preselected sampling)
    t_unique, idx_start, counts = np.unique(t_sorted, return_index=True, return_counts=True)
    # sum delta within group
    sums = np.add.reduceat(d_sorted, idx_start)

    # group proportions
    y = sums / counts
    w = counts.astype(float)

    # isotonic on proportions (equiv to GCM slopes / current-status MLE)
    f_star_group = _pava_non_decreasing(y, w)

    # transform back to F
    f_e = (f_star_group - (1 - r) / 2.0) / r
    f_hat = np.clip(f_e, 0.0, 1.0)

    return IsotonicCDF(t_unique=t_unique, fhat_at_t=f_hat)


def make_density_based_T(n: int, G_inv: Callable[[Array], Array], rng: np.random.Generator) -> Array:
    """Sample T_i i.i.d. from continuous CDF G via inverse-CDF."""
    u = rng.random(n)
    t = G_inv(u)
    # numerical safety
    return np.clip(t, 0.0, 1.0)


def make_preselected_T(n: int, grid: Array, probs: Array, rng: np.random.Generator) -> Array:
    """Sample T_i from a discrete distribution on grid points."""
    grid = np.asarray(grid, dtype=float)
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    idx = rng.choice(len(grid), size=n, replace=True, p=probs)
    return grid[idx]
