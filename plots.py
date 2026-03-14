"""project/plots.py

Paper-style plotting utilities for liu24z.pdf.

This script can generate the paper's Figures 1–4.

Figures in liu24z.pdf (see rendered PDF pages):
- Figure 1 (main, p.8): RCE (WMSE/κ) + coverage rate (2 stacked panels)
- Figure 2 (appendix, p.12): PDF (left) + CDF (right)
- Figure 3 (appendix, p.12): estimation vs truth (left) + absolute error (right)
- Figure 4 (appendix, p.13): SMAE (left) + MAE (right)

Notes
- The paper uses more sample sizes (up to 1e8 for Figure 1, up to 1e7 for Figure 4).
  If you pass smaller n_list (e.g. 1e3,1e4,1e5), the figure will match the layout/style
  but contain fewer points.

Examples
  # after running experiments_preselected.py
  python plots.py figure1 --results preselected_results.json --kappa 10 --out fig1.png

  # Figure 2 (no experiment outputs needed)
  python plots.py figure2 --out fig2.png

  # Figure 3 (simulation)
  python plots.py figure3 --n_list 1000,10000,100000 --r 0.5 --seed 1 --out fig3.png

  # after running experiments_density.py
  python plots.py figure4 --results density_results.json --out fig4.png
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from distributions import (
    continuous_bernoulli_dist,
    truncated_normal_dist,
    uniform_dist,
)
from ldp_cdf import (
    constrained_isotonic_cdf,
    ldp_random_response_indicator,
    make_density_based_T,
)
from distributions import uniform_G_inv


def _parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _dist_code(dist_name: str) -> str:
    """Short code used by paper legends."""
    if dist_name.startswith("U("):
        return "U"
    if dist_name.startswith("Nc("):
        return "Nc"
    if dist_name.startswith("CB("):
        return "CB"
    # fall back
    return dist_name


def _paper_line_style(dist_code: str) -> str:
    # Match Figures 1/4: U solid, CB dotted, Nc dashed.
    if dist_code == "U":
        return "-"
    if dist_code == "CB":
        return ":"
    if dist_code == "Nc":
        return "--"
    return "-"


def _color_for_r(r: float) -> str:
    # Match paper: r=0.25/0.5/0.9 share colors across distributions.
    if abs(r - 0.25) < 1e-12:
        return "C0"
    if abs(r - 0.5) < 1e-12:
        return "C1"
    if abs(r - 0.9) < 1e-12:
        return "C2"
    # fallback: deterministic mapping
    return f"C{int(round(r * 10)) % 10}"


# ---------------------- Figure 1 ----------------------


def figure1_paper(results_json: str, kappa: int, out: str):
    """Reproduce the layout/style of Figure 1 from preselected_results.json."""
    with open(results_json, "r", encoding="utf-8") as f:
        res = json.load(f)

    k = str(kappa)
    if k not in res:
        raise KeyError(f"kappa={kappa} not found in {results_json}")

    # Preserve the paper's dist ordering
    dist_names = [
        "U(0,1)",
        "Nc(0,1,μ=1/2,σ^2=1/4)",
        "CB(λ=0.25)",
    ]
    # Some JSONs may use a different order or omit items; keep only those present
    dist_names = [d for d in dist_names if d in res[k]]

    # r values in paper
    r_list = [0.25, 0.5, 0.9]

    # Collect n values (assume all dists share n keys)
    any_dist = dist_names[0]
    n_list = sorted(int(n) for n in res[k][any_dist].keys())

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(7.0, 7.0), sharex=True)

    # Top: RCE
    for dist in dist_names:
        code = _dist_code(dist)
        ls = _paper_line_style(code)
        for r in r_list:
            r_key = str(r)
            y = [res[k][dist][str(n)][r_key]["RCE_mean"] for n in n_list]
            ax_top.plot(
                n_list,
                y,
                linestyle=ls,
                color=_color_for_r(r),
                label=f"{code} r={r}",
            )
    ax_top.axhline(1.0, linestyle="--", linewidth=1)
    ax_top.set_xscale("log")
    ax_top.set_ylabel(r"$e_r$")
    ax_top.legend(ncol=3, fontsize=8, loc="lower right")

    # Bottom: Coverage
    for dist in dist_names:
        code = _dist_code(dist)
        ls = _paper_line_style(code)
        for r in r_list:
            r_key = str(r)
            y = [res[k][dist][str(n)][r_key]["coverage_rate"] for n in n_list]
            ax_bot.plot(
                n_list,
                y,
                linestyle=ls,
                color=_color_for_r(r),
                label=f"{code} r={r}",
            )
    ax_bot.axhline(0.95, linestyle="--", linewidth=1)
    ax_bot.set_xscale("log")
    ax_bot.set_xlabel("n")
    ax_bot.set_ylabel("Coverage rate")
    ax_bot.legend(ncol=3, fontsize=8, loc="upper center")

    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


# ---------------------- Figure 2 ----------------------


def figure2_paper(out: str, grid_size: int = 1001):
    """Reproduce Figure 2: PDF (left) and CDF (right) of the 3 distributions."""
    dists = [uniform_dist(), truncated_normal_dist(), continuous_bernoulli_dist(0.25)]
    x = np.linspace(0.0, 1.0, grid_size)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10.0, 4.0))

    for dist in dists:
        if dist.pdf is None:
            raise ValueError(f"{dist.name} has no pdf")
        ax_l.plot(x, dist.pdf(x), label=dist.paper_label)
    ax_l.set_xlabel("x")
    ax_l.set_ylabel("f(x)")
    ax_l.legend(loc="lower center", fontsize=9)

    for dist in dists:
        ax_r.plot(x, dist.cdf(x), label=dist.paper_label)
    ax_r.set_xlabel("x")
    ax_r.set_ylabel("F(x)")
    ax_r.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


# ---------------------- Figure 3 ----------------------


def figure3_paper(n_list: List[int], r: float, seed: int, out: str, grid_size: int = 2001):
    """Reproduce Figure 3 via simulation (uniform distribution, G uniform)."""
    rng = np.random.default_rng(seed)
    dist = uniform_dist()

    x_grid = np.linspace(0.0, 1.0, grid_size)
    ftrue = dist.cdf(x_grid)

    # Paper uses 4 n values with different linestyles
    linestyles = [":", "--", "-.", "-"]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10.0, 4.0))

    for i, n in enumerate(n_list):
        ls = linestyles[i % len(linestyles)]
        x = dist.sample(n, rng)
        t = make_density_based_T(n, uniform_G_inv, rng)
        delta = ldp_random_response_indicator(x, t, r, rng)
        cdf_hat = constrained_isotonic_cdf(delta, t, r)
        fhat = cdf_hat.eval_left(x_grid)

        ax_l.step(x_grid, fhat, where="post", linestyle=ls, label=f"n={n}")
        ax_r.plot(x_grid, np.abs(fhat - ftrue), linestyle=ls, label=f"n={n}")

    ax_l.plot(x_grid, ftrue, label="Truth")

    ax_l.set_xlabel("x")
    ax_l.set_ylabel(r"$\hat{F}(x)$")
    ax_l.legend(loc="upper left", fontsize=9)

    ax_r.set_xlabel("x")
    ax_r.set_ylabel(r"$|\hat{F}(x)-F(x)|$")
    ax_r.legend(loc="upper center", fontsize=9)

    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


# ---------------------- Figure 4 ----------------------


def figure4_paper(density_results_json: str, out: str):
    """Reproduce Figure 4 from density_results.json."""
    with open(density_results_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "results" not in data:
        raise KeyError("density_results.json must contain top-level key 'results'")

    # Preserve the paper's dist ordering
    dist_names = [
        "U(0,1)",
        "CB(λ=0.25)",
        "Nc(0,1,μ=1/2,σ^2=1/4)",
    ]
    dist_names = [d for d in dist_names if d in data["results"]]
    r_list = [0.25, 0.5, 0.9]

    # Collect n keys (assume all dists share them)
    any_dist = dist_names[0]
    n_list = sorted(int(n) for n in data["results"][any_dist].keys())

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10.0, 4.0))

    for dist in dist_names:
        code = _dist_code(dist)
        ls = _paper_line_style(code)
        for r in r_list:
            r_key = str(r)
            smae = [data["results"][dist][str(n)][r_key]["SMAE_mean"] for n in n_list]
            mae = [data["results"][dist][str(n)][r_key]["MAE_mean"] for n in n_list]

            ax_l.plot(
                n_list,
                smae,
                linestyle=ls,
                color=_color_for_r(r),
                label=f"{code} r={r}",
            )
            ax_r.plot(
                n_list,
                mae,
                linestyle=ls,
                color=_color_for_r(r),
                label=f"{code} r={r}",
            )

    ax_l.set_xscale("log")
    ax_r.set_xscale("log")

    ax_l.set_xlabel("n")
    ax_r.set_xlabel("n")

    ax_l.set_ylabel("SMAE")
    ax_r.set_ylabel("MAE")

    ax_l.legend(ncol=3, fontsize=8, loc="lower left")
    ax_r.legend(ncol=3, fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("figure1", help="Paper Figure 1 (RCE + coverage) from preselected_results.json")
    p1.add_argument("--results", type=str, required=True)
    p1.add_argument("--kappa", type=int, default=10)
    p1.add_argument("--out", type=str, default="fig1.png")

    p2 = sub.add_parser("figure2", help="Paper Figure 2 (PDF + CDF)")
    p2.add_argument("--out", type=str, default="fig2.png")
    p2.add_argument("--grid_size", type=int, default=1001)

    p3 = sub.add_parser("figure3", help="Paper Figure 3 (simulation)")
    p3.add_argument("--n_list", type=str, default="1000,10000,100000,1000000")
    p3.add_argument("--r", type=float, default=0.5)
    p3.add_argument("--seed", type=int, default=1)
    p3.add_argument("--out", type=str, default="fig3.png")
    p3.add_argument("--grid_size", type=int, default=2001)

    p4 = sub.add_parser("figure4", help="Paper Figure 4 (SMAE + MAE) from density_results.json")
    p4.add_argument("--results", type=str, required=True)
    p4.add_argument("--out", type=str, default="fig4.png")

    args = ap.parse_args()

    if args.cmd == "figure1":
        figure1_paper(args.results, args.kappa, args.out)
        print(f"Wrote {args.out}")
    elif args.cmd == "figure2":
        figure2_paper(args.out, grid_size=args.grid_size)
        print(f"Wrote {args.out}")
    elif args.cmd == "figure3":
        figure3_paper(_parse_int_list(args.n_list), args.r, args.seed, args.out, grid_size=args.grid_size)
        print(f"Wrote {args.out}")
    elif args.cmd == "figure4":
        figure4_paper(args.results, args.out)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
