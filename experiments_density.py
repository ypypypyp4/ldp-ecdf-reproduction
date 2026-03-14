
"""
Reproduce density-based sampling experiments (Table 1 + Figure 3 style plot).
Paper settings:
- Distributions: U(0,1), truncated normal, continuous Bernoulli (lambda=1/4)
- r in {0.25, 0.5, 0.9}
- n in {1e3, 1e4, 1e5, 1e6, 1e7}
- 10,000 replications (paper) - very heavy; here configurable.

Sampling G:
- "uniform": G(x)=x (uniform sampling)  —— main Table 1
- "oracle":  G=F (paper's appendix baseline, marked with *)

Metrics:
- MAE: max_x |Fhat(x) - F(x)| on a dense evaluation grid (left-continuous fill).
- L2: sqrt( ∫_0^1 (Fhat(x)-F(x))^2 dx ) approximated by trapezoid rule.
- SMAE: MAE * r * n^(1/3) / log(n) (paper definition)
"""

from __future__ import annotations
import argparse
import json
import numpy as np

from ldp_cdf import (
    constrained_isotonic_cdf,
    ldp_random_response_indicator,
    make_density_based_T,
)
from distributions import (
    uniform_dist,
    truncated_normal_dist,
    continuous_bernoulli_dist,
    uniform_G_inv,
)


def _trapz_compat(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoid-rule integration compatible across NumPy 1.x and 2.x.

    - NumPy 1.x: np.trapz exists (np.trapezoid may or may not exist)
    - NumPy 2.x: np.trapz removed; use np.trapezoid
    """
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def run_one_rep(dist, n: int, r: float, rng: np.random.Generator, eval_grid: np.ndarray, G_mode: str):
    x = dist.sample(n, rng)

    if G_mode == "uniform":
        G_inv = uniform_G_inv
    elif G_mode == "oracle":
        if dist.inv_cdf is None:
            raise ValueError(f"{dist.name} has no inv_cdf implementation")
        G_inv = dist.inv_cdf
    else:
        raise ValueError("G_mode must be 'uniform' or 'oracle'")

    t = make_density_based_T(n, G_inv, rng)
    delta = ldp_random_response_indicator(x, t, r, rng)
    cdf_hat = constrained_isotonic_cdf(delta, t, r)
    fhat = cdf_hat.eval_left(eval_grid)
    ftrue = dist.cdf(eval_grid)
    err = np.abs(fhat - ftrue)
    mae = float(err.max())
    l2 = float(np.sqrt(_trapz_compat((fhat - ftrue) ** 2, eval_grid)))
    smae = float(mae * (r * (n ** (1.0 / 3.0)) / np.log(max(n, 3))))
    return mae, l2, smae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_list", type=str, default="1000,10000,100000")
    ap.add_argument("--r_list", type=str, default="0.25,0.5,0.9")
    ap.add_argument("--G", type=str, default="uniform", choices=["uniform", "oracle"])
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--grid_size", type=int, default=2001)
    ap.add_argument("--out", type=str, default="density_results.json")
    args = ap.parse_args()

    n_list = [int(x) for x in args.n_list.split(",")]
    r_list = [float(x) for x in args.r_list.split(",")]
    eval_grid = np.linspace(0.0, 1.0, args.grid_size)

    dists = [uniform_dist(), truncated_normal_dist(), continuous_bernoulli_dist(0.25)]

    rng0 = np.random.default_rng(args.seed)
    results = {"G_mode": args.G, "results": {}}
    for dist in dists:
        dist_res = {}
        for n in n_list:
            n_res = {}
            for r in r_list:
                maes, l2s, smaes = [], [], []
                print(f"[{dist.name}] n={n} r={r} reps={args.reps}", flush=True)
                # independent reps (paper also reruns from scratch per n)
                for rep in range(args.reps):
                    if (rep + 1) % 200 == 0 or (rep + 1) == 1:
                        print(f"  rep {rep+1}/{args.reps}", flush=True)
                        
                    rng = np.random.default_rng(rng0.integers(0, 2**32 - 1))
                    mae, l2, smae = run_one_rep(dist, n, r, rng, eval_grid, args.G)
                    maes.append(mae); l2s.append(l2); smaes.append(smae)
                n_res[str(r)] = {
                    "MAE_mean": float(np.mean(maes)),
                    "MAE_std": float(np.std(maes, ddof=1)),
                    "L2_mean": float(np.mean(l2s)),
                    "L2_std": float(np.std(l2s, ddof=1)),
                    "SMAE_mean": float(np.mean(smaes)),
                    "SMAE_std": float(np.std(smaes, ddof=1)),
                }
            dist_res[str(n)] = n_res
        results["results"][dist.name] = dist_res

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
