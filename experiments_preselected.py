
"""
Reproduce preselected sampling inference experiments:
- Choose discrete observation times T on a grid of size kappa (κ = 10, 20, 30 in paper appendix tables).
- Compute WMSE and summarize Relative Chi-square Error (RCE) and coverage rate.

Important note:
The PDF typesets Eq.(6) with a leading sqrt(n), but the subsequent claim that WMSE ~ chi-square
and the appendix tables (RCE near 1, coverage near 0.95 for large n) are consistent with:
  WMSE = n * sum_j [ r^2 * w_j * (Fhat(x_j) - F(x_j))^2 / Var_num_j ],
where Var_num_j = (r F(x_j) + (1-r)/2) * ((1+r)/2 - r F(x_j)).
(Equivalently, WMSE = sum_j ( sqrt(n)*(Fhat-F)/sd_j )^2 .)

We implement this 'chi-square consistent' WMSE.
"""

from __future__ import annotations
import argparse
import json
import numpy as np
from scipy.stats import chi2

from ldp_cdf import (
    constrained_isotonic_cdf,
    ldp_random_response_indicator,
    make_preselected_T,
)
from distributions import (
    uniform_dist,
    truncated_normal_dist,
    continuous_bernoulli_dist,
    make_uniform_grid,
)


def wmse_chisq(
    fhat_at_grid: np.ndarray,
    ftrue_at_grid: np.ndarray,
    r: float,
    w: np.ndarray,
    n: int,
) -> float:
    var_num = (r * ftrue_at_grid + (1 - r) / 2.0) * ((1 + r) / 2.0 - r * ftrue_at_grid)
    # protect against numerical zeros
    var_num = np.maximum(var_num, 1e-15)
    return float(n * np.sum((r ** 2) * w * ((fhat_at_grid - ftrue_at_grid) ** 2) / var_num))


def wmse_literal_sqrtn(
    fhat_at_grid: np.ndarray,
    ftrue_at_grid: np.ndarray,
    r: float,
    w: np.ndarray,
    n: int,
) -> float:
    """Literal reading of the PDF typesetting of Eq.(6) (has a leading sqrt(n)).

    IMPORTANT: This quantity is *not* chi-square distributed (dimension mismatch),
    so the coverage comparison in the paper will not hold.
    """
    var_num = (r * ftrue_at_grid + (1 - r) / 2.0) * ((1 + r) / 2.0 - r * ftrue_at_grid)
    var_num = np.maximum(var_num, 1e-15)
    return float(np.sqrt(n) * np.sum((r ** 2) * w * ((fhat_at_grid - ftrue_at_grid) ** 2) / var_num))


def run_one_rep(dist, n: int, r: float, kappa: int, rng: np.random.Generator, wmse_mode: str):
    x = dist.sample(n, rng)
    grid, probs = make_uniform_grid(kappa)
    t = make_preselected_T(n, grid, probs, rng)
    delta = ldp_random_response_indicator(x, t, r, rng)
    cdf_hat = constrained_isotonic_cdf(delta, t, r)
    # evaluate only on grid points (they are included in unique t)
    fhat = cdf_hat.eval_left(grid)
    ftrue = dist.cdf(grid)

    # for uniform G, weights are probs=1/kappa (and correspond to G'(xj)-G'(xj-1) in paper)
    w = probs
    if wmse_mode == "chisq":
        wmse = wmse_chisq(fhat, ftrue, r, w, n)
    elif wmse_mode == "literal_sqrtn":
        wmse = wmse_literal_sqrtn(fhat, ftrue, r, w, n)
    else:
        raise ValueError("wmse_mode must be 'chisq' or 'literal_sqrtn'")
    rce = wmse / float(kappa)
    cover = float(wmse < chi2.ppf(0.95, df=kappa))
    return wmse, rce, cover


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_list", type=str, default="1000,10000,100000")
    ap.add_argument("--r_list", type=str, default="0.25,0.5,0.9")
    ap.add_argument("--kappa_list", type=str, default="10,20,30")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", type=str, default="preselected_results.json")
    ap.add_argument(
        "--wmse_mode",
        type=str,
        default="chisq",
        choices=["chisq", "literal_sqrtn"],
        help="WMSE definition: 'chisq' matches the paper tables; 'literal_sqrtn' matches PDF Eq.(6) typesetting.",
    )
    args = ap.parse_args()

    n_list = [int(x) for x in args.n_list.split(",")]
    r_list = [float(x) for x in args.r_list.split(",")]
    kappa_list = [int(x) for x in args.kappa_list.split(",")]

    dists = [uniform_dist(), truncated_normal_dist(), continuous_bernoulli_dist(0.25)]

    rng0 = np.random.default_rng(args.seed)
    results = {}
    for kappa in kappa_list:
        k_res = {}
        for dist in dists:
            dist_res = {}
            for n in n_list:
                n_res = {}
                for r in r_list:
                    rces, covers = [], []
                    for rep in range(args.reps):
                        rng = np.random.default_rng(rng0.integers(0, 2**32 - 1))
                        wmse, rce, cover = run_one_rep(dist, n, r, kappa, rng, args.wmse_mode)
                        rces.append(rce); covers.append(cover)
                    n_res[str(r)] = {
                        "coverage_rate": float(np.mean(covers)),
                        "RCE_mean": float(np.mean(rces)),
                        "RCE_std": float(np.std(rces, ddof=1)),
                    }
                dist_res[str(n)] = n_res
            k_res[dist.name] = dist_res
        results[str(kappa)] = k_res

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
