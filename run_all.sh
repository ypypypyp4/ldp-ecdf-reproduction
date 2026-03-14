#!/usr/bin/env bash
set -euo pipefail

# Paper uses r in {0.25, 0.5, 0.9}
R_LIST="0.25,0.5,0.9"

# -------------------- n lists --------------------
# Tables 1 & 3: n = 1e3 ... 1e7
N_LIST_PAPER_TABLES="1000,10000,100000,1000000,10000000"

# Figure 1 + Table 4: kappa=10, max n = 1e7
N_LIST_FIG1_K10="1000,10000,100000,1000000,10000000"

# Figure 4: reduced dense grid
N_LIST_FIG4_DENSE="1000,2000,5000,10000,20000,50000,100000,200000,500000,1000000,2000000,5000000,10000000"

# Tables 5 & 6: kappa=20,30, stop at 1e6
N_LIST_PRESELECTED_K2030="1000,10000,100000,1000000"

# -------------------- Replications --------------------
REPS_DENSITY=2000
REPS_PRESELECTED=2000
REPS_TABLE7=30

# ============================================================
# Density-based sampling (uniform G): Figure 4 + Table 1
# ============================================================

python experiments_density.py --G uniform \
  --n_list "$N_LIST_FIG4_DENSE" \
  --r_list "$R_LIST" \
  --reps "$REPS_DENSITY" \
  --out density_results_fig4_dense.json

python - <<'PY'
import json

paper_ns = {1000, 10000, 100000, 1000000, 10000000}

with open('density_results_fig4_dense.json', 'r', encoding='utf-8') as f:
    obj = json.load(f)

out = {"G_mode": obj.get("G_mode", "uniform"), "results": {}}
for dist, dist_res in obj["results"].items():
    out["results"][dist] = {
        str(n): dist_res[str(n)]
        for n in sorted(paper_ns)
        if str(n) in dist_res
    }

with open('density_results_table1.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

print('Wrote density_results_table1.json')
PY

python plots.py figure4 --results density_results_fig4_dense.json --out fig4.png
python tables.py table1 --results density_results_table1.json --out table1

# ============================================================
# Density-based sampling (oracle G=F): Table 3
# ============================================================

python experiments_density.py --G oracle \
  --n_list "$N_LIST_PAPER_TABLES" \
  --r_list "$R_LIST" \
  --reps "$REPS_DENSITY" \
  --out density_results_oracle.json

python tables.py table3 --results density_results_oracle.json --out table3

# ============================================================
# Preselected sampling: Figure 1 + Table 4
# ============================================================

python experiments_preselected.py \
  --n_list "$N_LIST_FIG1_K10" \
  --r_list "$R_LIST" \
  --kappa_list 10 \
  --reps "$REPS_PRESELECTED" \
  --out preselected_results_k10.json

python plots.py figure1 --results preselected_results_k10.json --kappa 10 --out fig1.png
python tables.py table4 --results preselected_results_k10.json --out table4

# ============================================================
# Preselected sampling: Tables 5–6
# ============================================================

python experiments_preselected.py \
  --n_list "$N_LIST_PRESELECTED_K2030" \
  --r_list "$R_LIST" \
  --kappa_list 20,30 \
  --reps "$REPS_PRESELECTED" \
  --out preselected_results_k2030.json

python tables.py table5 --results preselected_results_k2030.json --out table5
python tables.py table6 --results preselected_results_k2030.json --out table6

# ============================================================
# Figure 2 / Figure 3 / Table 2
# ============================================================

python plots.py figure2 --out fig2.png
python plots.py figure3 --n_list "1000,10000,100000,1000000" --r 0.5 --seed 1 --out fig3.png
python tables.py table2 --out table2

# ============================================================
# Table 7
# ============================================================

python tables.py table7 \
  --n_list "1000,10000,100000,1000000,10000000" \
  --reps "$REPS_TABLE7" \
  --out table7

echo "Done. Generated fig1.png fig2.png fig3.png fig4.png and table1-7 (.md/.tex)."