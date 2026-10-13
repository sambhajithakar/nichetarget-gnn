"""Evaluate Xenium target rankings against external evidence (no ground truth exists).

Evidence sets (fixed before any model output was inspected):
  * OT-breast: Open Targets Platform 26.09 overall association with breast
    carcinoma (MONDO_0004989) >= 0.3.
  * OT-clinical: target of at least one drug at any clinical stage in Open
    Targets (drugAndClinicalCandidates), any indication.
  * DepMap-breast: Chronos gene effect < -0.5 in >= 30% of the 53 breast
    cancer cell lines (DepMap 25Q2, Zenodo copy 17161166).
Candidates: all panel genes except the state read-out genes of that section.
Metrics per method: AUPRC (and prevalence), number of evidence genes in the
top 25 with a one-sided hypergeometric p-value, and Spearman correlation of
the score with the continuous Open Targets association score. Replicate
concordance: Spearman correlation of scores between Rep1 and Rep2 on the
shared candidates and overlap of the top 25.

    python scripts/evaluate_xenium.py --runs results/xenium_rep1 results/xenium_rep2 \
        --ot-assoc .../breast_carcinoma_associations.csv --ot-drugs .../panel_druggability.csv \
        --depmap .../panel_breast_gene_effect.csv --out results/xenium_evaluation
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from math import comb

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nichetarget.np_graph import average_precision, spearman  # noqa: E402

TOPK = 25
META = {"gene", "is_readout", "in_lr_database", "best_niche"}


def hypergeom_sf(k, N, K, n):
    """P(X >= k) for X ~ Hypergeometric(population N, successes K, draws n)."""
    return float(sum(comb(K, i) * comb(N - K, n - i) for i in range(k, min(K, n) + 1)) / comb(N, n))


def method_columns(df):
    return [c for c in df.columns if c not in META and not c.startswith(("mean_expr", "efficacy_niche"))]


def evaluate(df, evidence, ot_score):
    cand = df[~df.is_readout].copy()
    rows = []
    for m in method_columns(df):
        s = cand[m].to_numpy(float)
        order = np.argsort(-s, kind="mergesort")
        top = cand.gene.to_numpy()[order[:TOPK]]
        row = {"method": m}
        for name, pos in evidence.items():
            y = cand.gene.isin(pos).to_numpy()
            k = int(np.isin(top, list(pos)).sum())
            row[f"{name}_AUPRC"] = average_precision(y, s)
            row[f"{name}_prevalence"] = float(y.mean())
            row[f"{name}_top{TOPK}"] = k
            row[f"{name}_top{TOPK}_p"] = hypergeom_sf(k, len(y), int(y.sum()), TOPK)
        row["spearman_with_OT_score"] = spearman(s, cand.gene.map(ot_score).fillna(0).to_numpy())
        row[f"top{TOPK}"] = ";".join(top)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--ot-assoc", required=True)
    ap.add_argument("--ot-drugs", required=True)
    ap.add_argument("--depmap", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ot = pd.read_csv(a.ot_assoc)
    ot_score = dict(zip(ot.symbol, ot.score))
    dr = pd.read_csv(a.ot_drugs)
    dm = pd.read_csv(a.depmap)
    evidence = {
        "OT_breast": set(ot.symbol[ot.score >= 0.3]),
        "OT_clinical": set(dr.symbol[dr.max_clinical_stage.notna()]),
        "DepMap_breast": set(dm.gene[dm.frac_lines_dependent_breast >= 0.3]),
    }
    summary = {"evidence_sizes_in_panel": {}}
    tables = {}
    for run in a.runs:
        df = pd.read_csv(os.path.join(run, "gene_scores.csv"))
        name = os.path.basename(run.rstrip("/"))
        res = evaluate(df, evidence, ot_score)
        res.insert(0, "section", name)
        tables[name] = (df, res)
        res.to_csv(os.path.join(a.out, f"evaluation_{name}.csv"), index=False)
        cand = df[~df.is_readout]
        summary["evidence_sizes_in_panel"][name] = {k: int(cand.gene.isin(v).sum()) for k, v in evidence.items()}
        summary["evidence_sizes_in_panel"][name]["n_candidates"] = int(len(cand))
    if len(tables) >= 2:
        (n1, (d1, _)), (n2, (d2, _)) = list(tables.items())[:2]
        common = sorted(set(d1.gene[~d1.is_readout]) & set(d2.gene[~d2.is_readout]))
        conc = []
        for m in method_columns(d1):
            if m not in d2.columns:
                continue
            s1 = d1.set_index("gene").loc[common, m].to_numpy(float)
            s2 = d2.set_index("gene").loc[common, m].to_numpy(float)
            t1 = set(np.array(common)[np.argsort(-s1, kind="mergesort")[:TOPK]])
            t2 = set(np.array(common)[np.argsort(-s2, kind="mergesort")[:TOPK]])
            conc.append({"method": m, "n_common_candidates": len(common), "spearman_rep1_rep2": spearman(s1, s2),
                         f"top{TOPK}_overlap": len(t1 & t2), "shared_top_genes": ";".join(sorted(t1 & t2))})
        pd.DataFrame(conc).to_csv(os.path.join(a.out, "replicate_concordance.csv"), index=False)
    pd.concat([t[1] for t in tables.values()]).to_csv(os.path.join(a.out, "evaluation_all.csv"), index=False)
    summary["evidence_sets"] = {k: sorted(v) for k, v in evidence.items()}
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    pd.set_option("display.width", 250)
    for n, (_, r) in tables.items():
        print(n)
        print(r.drop(columns=[f"top{TOPK}"]).round(3).to_string())


if __name__ == "__main__":
    main()
