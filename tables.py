"""project/tables.py

Generate paper-style tables (Table 1–7) for liu24z.pdf.

This script formats experiment outputs into the same *cell content* as the paper.
It does not fabricate missing rows: if your results JSON only contains
n={1000,10000,100000}, then the tables will only include those n values.

Outputs
- Always writes BOTH Markdown (.md) and LaTeX (.tex).
- `--out` can be a base name (e.g., table1) or a path ending with .md or .tex.

Commands
  - table1 : from density_results.json (uniform sampling, G(x)=x)
  - table3 : from density_results_oracle.json (oracle sampling, G=F)
  - table2 : conversion table between r and epsilon
  - table4 : kappa=10 preselected sampling (coverage(RCE))
  - table5 : kappa=20 preselected sampling (coverage(RCE))
  - table6 : kappa=30 preselected sampling (coverage(RCE))
  - table7 : runtime benchmark for constrained_isotonic_cdf
  - table456 : write table4/5/6 in one command (optional convenience)

Examples
  python tables.py table1 --results density_results.json --out table1
  python tables.py table4 --results preselected_results.json --out table4
  python tables.py table7 --n_list 1000,10000,100000 --reps 30 --out table7
"""

from __future__ import annotations

import argparse
import json
import time
from typing import List, Tuple

import numpy as np

from ldp_cdf import constrained_isotonic_cdf, epsilon_from_r


# ------------------------------- helpers -------------------------------

def _parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _dist_order_key(name: str) -> int:
    # Match paper tables: U, Nc, CB
    if name.startswith("U("):
        return 0
    if name.startswith("Nc("):
        return 1
    if name.startswith("CB("):
        return 2
    return 99


def _fmt_cell(a: float, b: float) -> str:
    return f"{a:.3f}({b:.3f})"


def _out_pair(out: str) -> Tuple[str, str]:
    """Given an --out argument, return (md_path, tex_path)."""
    o = out.strip()
    lo = o.lower()
    if lo.endswith(".md"):
        base = o[:-3]
    elif lo.endswith(".tex"):
        base = o[:-4]
    else:
        base = o
    return base + ".md", base + ".tex"


def _write_both(out: str, md_text: str, tex_text: str) -> Tuple[str, str]:
    md_path, tex_path = _out_pair(out)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex_text)
    return md_path, tex_path


def _latex_math(s: str) -> str:
    """Convert a distribution name into a LaTeX-ish math string."""
    rep = (
        ("μ", r"\\mu"),
        ("σ", r"\\sigma"),
        ("λ", r"\\lambda"),
        ("^2", r"^{2}"),
    )
    out = s
    for a, b in rep:
        out = out.replace(a, b)
    return f"${out}$"


# ------------------------------ Table 1 / 3 ------------------------------

def density_table_md(results_path: str, title: str) -> str:
    with open(results_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if "results" not in obj:
        raise ValueError("density results JSON must have top-level key 'results'")

    results = obj["results"]
    dists = sorted(results.keys(), key=_dist_order_key)
    n_list = sorted(int(n) for n in next(iter(results.values())).keys())
    r_list = sorted(float(r) for r in next(iter(next(iter(results.values())).values())).keys())

    header = f"# {title}\n\n"
    header += "| n | r | " + " | ".join(dists) + " |\n"
    header += "|---:|---:|" + "|".join([":---:" for _ in dists]) + "|\n"

    rows = []
    for n in n_list:
        for r in r_list:
            cells = []
            for dist in dists:
                m = results[dist][str(n)][str(r)]
                cells.append(_fmt_cell(m["MAE_mean"], m["L2_mean"]))
            rows.append(f"| {n} | {r:g} | " + " | ".join(cells) + " |")

    return header + "\n".join(rows) + "\n"


def density_table_tex(results_path: str, caption: str) -> str:
    with open(results_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if "results" not in obj:
        raise ValueError("density results JSON must have top-level key 'results'")

    results = obj["results"]
    dists = sorted(results.keys(), key=_dist_order_key)
    n_list = sorted(int(n) for n in next(iter(results.values())).keys())
    r_list = sorted(float(r) for r in next(iter(next(iter(results.values())).values())).keys())

    cols = "rr" + "c" * len(dists)
    lines = [
        r"\\begin{table}[t]",
        r"\\centering",
        rf"\\caption{{{caption}}}",
        rf"\\begin{{tabular}}{{{cols}}}",
        r"\\toprule",
        r"$n$ & $r$ & " + " & ".join(_latex_math(d) for d in dists) + r" \\",
        r"\\midrule",
    ]

    for n in n_list:
        for r in r_list:
            cells = []
            for dist in dists:
                m = results[dist][str(n)][str(r)]
                cells.append(_fmt_cell(m["MAE_mean"], m["L2_mean"]))
            lines.append(f"{n} & {r:g} & " + " & ".join(cells) + r" \\")

    lines += [r"\\bottomrule", r"\\end{tabular}", r"\\end{table}", ""]
    return "\n".join(lines)


# ------------------------------ Table 2 ------------------------------

def conversion_table_md() -> str:
    left = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    right = [0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    lines = ["# Table 2. Conversion table between r and ε\n"]
    lines.append("| r | ε | r | ε |\n|---:|---:|---:|---:|\n")
    for a, b in zip(left, right):
        ea = 0.0 if a == 0.0 else epsilon_from_r(a)
        eb = epsilon_from_r(b)
        lines.append(f"| {a:g} | {ea:.2f} | {b:g} | {eb:.2f} |\n")
    return "".join(lines)


def conversion_table_tex() -> str:
    left = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    right = [0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    lines = [
        r"\\begin{table}[t]",
        r"\\centering",
        r"\\caption{Table 2. Conversion table between $r$ and $\\varepsilon$}",
        r"\\begin{tabular}{rrrr}",
        r"\\toprule",
        r"$r$ & $\\varepsilon$ & $r$ & $\\varepsilon$ \\",
        r"\\midrule",
    ]

    for a, b in zip(left, right):
        ea = 0.0 if a == 0.0 else epsilon_from_r(a)
        eb = epsilon_from_r(b)
        lines.append(f"{a:g} & {ea:.2f} & {b:g} & {eb:.2f} \\")

    lines += [r"\\bottomrule", r"\\end{tabular}", r"\\end{table}", ""]
    return "\n".join(lines)


# ------------------------------ Table 4 / 5 / 6 ------------------------------

def preselected_table_md(results_path: str, kappa: int) -> str:
    with open(results_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    k = str(kappa)
    if k not in obj:
        raise ValueError(f"preselected results JSON has no kappa={kappa}")

    results = obj[k]
    dists = sorted(results.keys(), key=_dist_order_key)
    n_list = sorted(int(n) for n in next(iter(results.values())).keys())
    r_list = sorted(float(r) for r in next(iter(next(iter(results.values())).values())).keys())

    table_no = {10: 4, 20: 5, 30: 6}.get(kappa, "?")
    header = f"# Table {table_no}. Empirical coverage rate (RCE) with κ={kappa}\n\n"
    header += "| n | r | " + " | ".join(dists) + " |\n"
    header += "|---:|---:|" + "|".join([":---:" for _ in dists]) + "|\n"

    rows = []
    for n in n_list:
        for r in r_list:
            cells = []
            for dist in dists:
                m = results[dist][str(n)][str(r)]
                cells.append(_fmt_cell(m["coverage_rate"], m["RCE_mean"]))
            rows.append(f"| {n} | {r:g} | " + " | ".join(cells) + " |")

    return header + "\n".join(rows) + "\n"


def preselected_table_tex(results_path: str, kappa: int) -> str:
    with open(results_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    k = str(kappa)
    if k not in obj:
        raise ValueError(f"preselected results JSON has no kappa={kappa}")

    results = obj[k]
    dists = sorted(results.keys(), key=_dist_order_key)
    n_list = sorted(int(n) for n in next(iter(results.values())).keys())
    r_list = sorted(float(r) for r in next(iter(next(iter(results.values())).values())).keys())

    table_no = {10: 4, 20: 5, 30: 6}.get(kappa, None)
    caption = (
        f"Table {table_no}. Empirical coverage rate (RCE) with $\\kappa={kappa}$"
        if table_no is not None
        else f"Empirical coverage rate (RCE) with $\\kappa={kappa}$"
    )

    cols = "rr" + "c" * len(dists)
    lines = [
        r"\\begin{table}[t]",
        r"\\centering",
        rf"\\caption{{{caption}}}",
        rf"\\begin{{tabular}}{{{cols}}}",
        r"\\toprule",
        r"$n$ & $r$ & " + " & ".join(_latex_math(d) for d in dists) + r" \\",
        r"\\midrule",
    ]

    for n in n_list:
        for r in r_list:
            cells = []
            for dist in dists:
                m = results[dist][str(n)][str(r)]
                cells.append(_fmt_cell(m["coverage_rate"], m["RCE_mean"]))
            lines.append(f"{n} & {r:g} & " + " & ".join(cells) + r" \\")

    lines += [r"\\bottomrule", r"\\end{tabular}", r"\\end{table}", ""]
    return "\n".join(lines)


# ------------------------------ Table 7 ------------------------------

def benchmark_table_md(n_list: List[int], reps: int, seed: int, r: float = 0.5) -> str:
    rng = np.random.default_rng(seed)

    means_ms: List[float] = []
    stds_ms: List[float] = []

    for n in n_list:
        times = []
        for _ in range(reps):
            t = rng.random(n, dtype=float)
            delta = (rng.random(n) < 0.5).astype(np.int8)
            t0 = time.perf_counter()
            _ = constrained_isotonic_cdf(delta, t, r)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
        means_ms.append(float(np.mean(times)))
        stds_ms.append(float(np.std(times, ddof=1)))

    header = "# Table 7. Computation times (ms) and standard deviation\n\n"
    header += "| metric | " + " | ".join([str(n) for n in n_list]) + " |\n"
    header += "|:--|" + "|".join([":---:" for _ in n_list]) + "|\n"

    row_mean = "| average time (ms) | " + " | ".join([f"{m:.3g}" for m in means_ms]) + " |\n"
    row_std = "| standard deviation | " + " | ".join([f"{s:.3g}" for s in stds_ms]) + " |\n"
    return header + row_mean + row_std


def benchmark_table_tex(n_list: List[int], reps: int, seed: int, r: float = 0.5) -> str:
    rng = np.random.default_rng(seed)

    means_ms: List[float] = []
    stds_ms: List[float] = []

    for n in n_list:
        times = []
        for _ in range(reps):
            t = rng.random(n, dtype=float)
            delta = (rng.random(n) < 0.5).astype(np.int8)
            t0 = time.perf_counter()
            _ = constrained_isotonic_cdf(delta, t, r)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
        means_ms.append(float(np.mean(times)))
        stds_ms.append(float(np.std(times, ddof=1)))

    cols = "l" + "c" * len(n_list)
    lines = [
        r"\\begin{table}[t]",
        r"\\centering",
        r"\\caption{Table 7. Computation times (ms) and standard deviation}",
        rf"\\begin{{tabular}}{{{cols}}}",
        r"\\toprule",
        r"metric & " + " & ".join(str(n) for n in n_list) + r" \\",
        r"\\midrule",
        r"average time (ms) & " + " & ".join(f"{m:.3g}" for m in means_ms) + r" \\",
        r"standard deviation & " + " & ".join(f"{s:.3g}" for s in stds_ms) + r" \\",
        r"\\bottomrule",
        r"\\end{tabular}",
        r"\\end{table}",
        "",
    ]
    return "\n".join(lines)


# -------------------------------- CLI --------------------------------

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("table1", help="Paper Table 1 from density_results.json (uniform sampling)")
    p1.add_argument("--results", type=str, required=True)
    p1.add_argument("--out", type=str, default="table1")

    p3 = sub.add_parser("table3", help="Paper Table 3 from density_results.json (oracle sampling, G=F)")
    p3.add_argument("--results", type=str, required=True)
    p3.add_argument("--out", type=str, default="table3")

    p2 = sub.add_parser("table2", help="Paper Table 2 conversion between r and epsilon")
    p2.add_argument("--out", type=str, default="table2")

    p4 = sub.add_parser("table4", help="Paper Table 4 (kappa=10) from preselected_results.json")
    p4.add_argument("--results", type=str, required=True)
    p4.add_argument("--out", type=str, default="table4")

    p5 = sub.add_parser("table5", help="Paper Table 5 (kappa=20) from preselected_results.json")
    p5.add_argument("--results", type=str, required=True)
    p5.add_argument("--out", type=str, default="table5")

    p6 = sub.add_parser("table6", help="Paper Table 6 (kappa=30) from preselected_results.json")
    p6.add_argument("--results", type=str, required=True)
    p6.add_argument("--out", type=str, default="table6")

    p456 = sub.add_parser("table456", help="Write table4/table5/table6 (kappa=10/20/30) in one command")
    p456.add_argument("--results", type=str, required=True)
    p456.add_argument("--out_prefix", type=str, default="table")

    p7 = sub.add_parser("table7", help="Paper Table 7 benchmark (runtime)")
    p7.add_argument("--n_list", type=str, default="1000,10000,100000")
    p7.add_argument("--reps", type=int, default=30)
    p7.add_argument("--seed", type=int, default=123)
    p7.add_argument("--r", type=float, default=0.5)
    p7.add_argument("--out", type=str, default="table7")

    args = ap.parse_args()

    if args.cmd == "table1":
        title = "Table 1. Empirical results of uniform consistency (L2 consistency) under uniform sampling"
        md = density_table_md(args.results, title)
        tex = density_table_tex(args.results, title)
        md_path, tex_path = _write_both(args.out, md, tex)
        print(f"Wrote {md_path} and {tex_path}")

    elif args.cmd == "table3":
        title = "Table 3. Empirical results of uniform consistency (L2 consistency) under G = F"
        md = density_table_md(args.results, title)
        tex = density_table_tex(args.results, title)
        md_path, tex_path = _write_both(args.out, md, tex)
        print(f"Wrote {md_path} and {tex_path}")

    elif args.cmd == "table2":
        md = conversion_table_md()
        tex = conversion_table_tex()
        md_path, tex_path = _write_both(args.out, md, tex)
        print(f"Wrote {md_path} and {tex_path}")

    elif args.cmd == "table4":
        md = preselected_table_md(args.results, 10)
        tex = preselected_table_tex(args.results, 10)
        md_path, tex_path = _write_both(args.out, md, tex)
        print(f"Wrote {md_path} and {tex_path}")

    elif args.cmd == "table5":
        md = preselected_table_md(args.results, 20)
        tex = preselected_table_tex(args.results, 20)
        md_path, tex_path = _write_both(args.out, md, tex)
        print(f"Wrote {md_path} and {tex_path}")

    elif args.cmd == "table6":
        md = preselected_table_md(args.results, 30)
        tex = preselected_table_tex(args.results, 30)
        md_path, tex_path = _write_both(args.out, md, tex)
        print(f"Wrote {md_path} and {tex_path}")

    elif args.cmd == "table456":
        for kappa, no in [(10, 4), (20, 5), (30, 6)]:
            out = f"{args.out_prefix}{no}"
            md = preselected_table_md(args.results, kappa)
            tex = preselected_table_tex(args.results, kappa)
            md_path, tex_path = _write_both(out, md, tex)
            print(f"Wrote {md_path} and {tex_path}")

    elif args.cmd == "table7":
        n_list = _parse_int_list(args.n_list)
        md = benchmark_table_md(n_list, args.reps, args.seed, r=args.r)
        tex = benchmark_table_tex(n_list, args.reps, args.seed, r=args.r)
        md_path, tex_path = _write_both(args.out, md, tex)
        print(f"Wrote {md_path} and {tex_path}")


if __name__ == "__main__":
    main()
