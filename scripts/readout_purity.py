"""Exploratory (post hoc) diagnostic for the cross-patient replication: is the read-out programme malignant?

For each section, a read-out gene counts as malignant-specific when its mean expression is highest in the
malignant lineage (mean_expr_<lineage> columns of gene_scores.csv). Purity = fraction of the 15 read-out
genes that are malignant-specific. Not part of PRESPECIFIED.md; reported as exploratory.

    python scripts/readout_purity.py --discovery results/xenium_rep1 --runs results/replication/* \
        --per-section results/replication/summary/replication_per_section.csv --out results/replication/summary
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def purity(run):
    d = pd.read_csv(os.path.join(run, "gene_scores.csv")).set_index("gene")
    cols = [c for c in d.columns if c.startswith("mean_expr_") and c != "mean_expr_target"]
    ro = d[d.is_readout][cols].fillna(-np.inf)
    best = ro.idxmax(axis=1)
    info = json.load(open(os.path.join(run, "run_info.json")))
    return float((best == "mean_expr_malignant").mean()), info.get("readout_R2_model0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--discovery", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--per-section", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for run in [a.discovery] + sorted(a.runs):
        if not os.path.exists(os.path.join(run, "run_info.json")):
            continue
        p, r2 = purity(run)
        rows.append({"section": os.path.basename(run.rstrip("/")), "readout_purity": p, "readout_R2": r2})
    pur = pd.DataFrame(rows)
    ps = pd.read_csv(a.per_section)
    m = ps[ps.method == "GNN"].merge(pur, on="section")
    prim = m[m.passed_gate & ~m.section.isin(["B1_2", "B1_3"])]
    rho, p = spearmanr(prim.readout_purity, prim.vs_disc_top25_overlap)
    lines = [f"donors={len(prim)}  Spearman(purity, GNN top-25 overlap with discovery)={rho:.3f}  p={p:.4f}"]
    for lab, sub in [("purity>=0.5", prim[prim.readout_purity >= 0.5]), ("purity<0.5", prim[prim.readout_purity < 0.5])]:
        lines.append(f"{lab}: n={len(sub)} median overlap={sub.vs_disc_top25_overlap.median()} "
                     f"median spearman={sub.vs_disc_spearman.median():.3f} sections={','.join(sub.section)}")
    os.makedirs(a.out, exist_ok=True)
    pur.merge(m[["section", "vs_disc_spearman", "vs_disc_top25_overlap", "passed_gate"]], on="section", how="left") \
        .to_csv(os.path.join(a.out, "readout_purity.csv"), index=False)
    open(os.path.join(a.out, "readout_purity.txt"), "w").write("\n".join(lines) + "\n")
    print(pur.round(3).to_string()); print("\n".join(lines))


if __name__ == "__main__":
    main()
