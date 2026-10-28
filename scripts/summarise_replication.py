"""Cross-patient replication of the Xenium rankings (pre-specified in results/replication/PRESPECIFIED.md).

    python scripts/summarise_replication.py --discovery results/xenium_rep1 --runs results/replication/* \
        --ot-assoc .../breast_carcinoma_associations.csv --ot-drugs .../panel_druggability.csv \
        --depmap .../panel_breast_gene_effect.csv --out results/replication/summary
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nichetarget.np_graph import average_precision, spearman  # noqa: E402
from evaluate_xenium import hypergeom_sf  # noqa: E402

TOPK = 25
MIN_MALIGNANT, MIN_STATE = 2000, 500
METHODS = {"GNN": "NicheTarget-GNN (single model)", "LR-niche": "LR-communication (niche-stratified)",
           "DE-niche": "DE (niche-stratified)", "DE": "DE", "NbrCorr-niche": "NeighbourhoodCorr (niche-stratified)"}
DISC_ENSEMBLE = "NicheTarget-GNN (ensemble of 3)"
PRIMARY_EXCLUDE = {"B1_2", "B1_3"}      # within-tumour technical replicates of B1_1


def top(genes, s, k=TOPK):
    return set(np.asarray(genes)[np.argsort(-np.asarray(s, float), kind="mergesort")[:k]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--discovery", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--ot-assoc", required=True)
    ap.add_argument("--ot-drugs", required=True)
    ap.add_argument("--depmap", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ot = pd.read_csv(a.ot_assoc); dr = pd.read_csv(a.ot_drugs); dm = pd.read_csv(a.depmap)
    evidence = {"OT_breast": set(ot.symbol[ot.score >= 0.3]),
                "OT_clinical": set(dr.symbol[dr.max_clinical_stage.notna()]),
                "DepMap_breast": set(dm.gene[dm.frac_lines_dependent_breast >= 0.3])}
    d0 = pd.read_csv(os.path.join(a.discovery, "gene_scores.csv")).set_index("gene")
    disc_cand = set(d0.index[~d0.is_readout])
    rows, qc = [], []
    for run in sorted(a.runs):
        p = os.path.join(run, "gene_scores.csv")
        if not os.path.exists(p):
            continue
        name = os.path.basename(run.rstrip("/"))
        info = json.load(open(os.path.join(run, "run_info.json")))
        n_mal = info["lineage_counts"]["malignant"]
        passed = n_mal >= MIN_MALIGNANT and info["n_target"] >= MIN_STATE and info["n_reference"] >= MIN_STATE
        d = pd.read_csv(p).set_index("gene")
        X_ok = {g for g in d.index}
        qc.append({"section": name, "n_cells": info["n_cells"], "n_malignant": n_mal, "n_target": info["n_target"],
                   "n_reference": info["n_reference"], "readout": ";".join(info["readout"]), "passed_gate": passed,
                   "mean_expr_ESR1_malignant": d.loc["ESR1", "mean_expr_malignant"] if "ESR1" in d.index else np.nan,
                   "mean_expr_PGR_malignant": d.loc["PGR", "mean_expr_malignant"] if "PGR" in d.index else np.nan,
                   "mean_expr_ERBB2_malignant": d.loc["ERBB2", "mean_expr_malignant"] if "ERBB2" in d.index else np.nan})
        common = sorted(disc_cand & set(d.index[~d.is_readout]) & X_ok)
        for short, col in METHODS.items():
            s = d.loc[common, col].to_numpy(float)
            refcols = [("vs_disc", col)]
            if short == "GNN" and DISC_ENSEMBLE in d0.columns:   # sensitivity discovery run is a single model
                refcols.append(("vs_disc_ensemble", DISC_ENSEMBLE))
            row = {"section": name, "method": short, "passed_gate": passed, "n_common": len(common)}
            for lab, rc in refcols:
                s0 = d0.loc[common, rc].to_numpy(float)
                ov = len(top(common, s) & top(common, s0))
                row[f"{lab}_spearman"] = spearman(s, s0)
                row[f"{lab}_top{TOPK}_overlap"] = ov
                row[f"{lab}_top{TOPK}_p"] = hypergeom_sf(ov, len(common), TOPK, TOPK)
            for ev, pos in evidence.items():
                y = np.isin(common, list(pos))
                k = len(top(common, s) & pos)
                row[f"{ev}_AUPRC"] = average_precision(y, s)
                row[f"{ev}_prevalence"] = float(y.mean())
                row[f"{ev}_top{TOPK}"] = k
                row[f"{ev}_top{TOPK}_p"] = hypergeom_sf(k, len(y), int(y.sum()), TOPK)
            row[f"top{TOPK}"] = ";".join(sorted(top(common, s)))
            rows.append(row)
    res = pd.DataFrame(rows); qcd = pd.DataFrame(qc)
    res.to_csv(os.path.join(a.out, "replication_per_section.csv"), index=False)
    qcd.to_csv(os.path.join(a.out, "replication_qc.csv"), index=False)
    prim = res[res.passed_gate & ~res.section.isin(PRIMARY_EXCLUDE)]
    agg = []
    for m, g in prim.groupby("method"):
        r = {"method": m, "n_donors": len(g)}
        for c in [c for c in g.columns if c.endswith(("_spearman", "_overlap", "_AUPRC"))]:
            r[c + "_median"] = float(g[c].median())
            r[c + "_q1"] = float(g[c].quantile(0.25)); r[c + "_q3"] = float(g[c].quantile(0.75))
        for ev in evidence:
            wins = int((g[f"{ev}_AUPRC"] > g[f"{ev}_prevalence"]).sum())
            r[f"{ev}_donors_AUPRC_above_prevalence"] = wins
            r[f"{ev}_sign_test_p_one_sided"] = hypergeom_free_sign(wins, len(g))
        agg.append(r)
    pd.DataFrame(agg).to_csv(os.path.join(a.out, "replication_across_donors.csv"), index=False)
    # gene-level recurrence of the GNN top 25 across donors
    gnn = prim[prim.method == "GNN"]
    cnt = pd.Series([g for t in gnn[f"top{TOPK}"] for g in t.split(";")]).value_counts()
    cnt.rename("n_donors_in_top25").to_csv(os.path.join(a.out, "gnn_top25_recurrence.csv"))
    print(qcd.to_string()); print(pd.DataFrame(agg).T.to_string()); print(cnt.head(25).to_string())


def hypergeom_free_sign(k, n):
    """One-sided exact sign test P(X >= k), X ~ Binomial(n, 0.5)."""
    from math import comb
    return float(sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)


if __name__ == "__main__":
    main()
