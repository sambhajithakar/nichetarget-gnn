"""Tie audit of the cross-patient top-25 overlaps (added 7 Nov 2026, after the revision results were written).

summarise_replication.top() breaks ties by input order (genes sorted alphabetically). When a method gives many
candidates the same score, the top-25 list is partly filled by alphabetical order, which is identical in every
section and inflates the overlap. Two cases were found:
  * LR-communication gives zero to every gene outside the curated pairs; with 180 genes fewer than 25 genes
    have a non-zero score.
  * Under --readout-rule malignant_specific, run_xenium.py stored the masked DE vector (-inf for genes that are
    not malignant-specific) as the "DE" baseline, so DE ranked only the malignant-specific genes.
This script reports, per section and method: the number of candidates with a distinct score above the tie block,
the original overlap, the tie-aware expected overlap (ties broken at random, 1,000 draws) and its chance
expectation; and, for the malignant_specific rule, DE recomputed without the mask (taken from the default-rule
run of the same section, which uses the same cells and states).

    python scripts/audit_ties.py --discovery results/revision/rep1_g180/{rule}_s1 \
        --runs results/revision/external/{rule}/* [--unmasked-de-from results/revision/external/de] \
        [--unmasked-de-disc results/revision/rep1_g180/de_s1] --out file.csv
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

TOPK = 25
METHODS = {"GNN": "NicheTarget-GNN (single model)", "LR-niche": "LR-communication (niche-stratified)",
           "DE-niche": "DE (niche-stratified)", "DE": "DE", "NbrCorr-niche": "NeighbourhoodCorr (niche-stratified)"}


def top_random(s, rng):
    jitter = rng.random(len(s))
    return set(np.lexsort((jitter, -s))[:TOPK])


def top_order(s):
    return set(np.argsort(-s, kind="mergesort")[:TOPK])


def n_above_tie(s):
    """Number of candidates strictly above the score of the TOPK-th candidate (ranks not decided by ties)."""
    kth = np.sort(s)[::-1][TOPK - 1]
    return int((s > kth).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--discovery", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--unmasked-de-from", default=None)
    ap.add_argument("--unmasked-de-disc", default=None)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    d0 = pd.read_csv(os.path.join(a.discovery, "gene_scores.csv")).set_index("gene")
    if a.unmasked_de_disc:
        d0["DE_unmasked"] = pd.read_csv(os.path.join(a.unmasked_de_disc, "gene_scores.csv")).set_index("gene").DE
    rows = []
    for run in sorted(a.runs):
        p = os.path.join(run, "gene_scores.csv")
        if not os.path.exists(p):
            continue
        name = os.path.basename(run.rstrip("/"))
        d = pd.read_csv(p).set_index("gene")
        meths = dict(METHODS)
        if a.unmasked_de_from:
            d["DE_unmasked"] = pd.read_csv(os.path.join(a.unmasked_de_from, name, "gene_scores.csv")).set_index("gene").DE
            meths["DE-unmasked"] = "DE_unmasked"
        common = sorted(set(d0.index[~d0.is_readout]) & set(d.index[~d.is_readout]))
        n = len(common)
        for short, col in meths.items():
            s = np.nan_to_num(d.loc[common, col].to_numpy(float), neginf=-1e9)
            s0 = np.nan_to_num(d0.loc[common, col].to_numpy(float), neginf=-1e9)
            ov_order = len(top_order(s) & top_order(s0))
            ov_rand = np.mean([len(top_random(s, rng) & top_random(s0, rng)) for _ in range(a.draws)])
            rows.append({"section": name, "method": short, "n_common": n, "expected_chance": TOPK * TOPK / n,
                         "n_decided_section": n_above_tie(s), "n_decided_discovery": n_above_tie(s0),
                         "overlap_original": ov_order, "overlap_tie_aware": round(float(ov_rand), 2)})
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    print(out.groupby("method")[["n_decided_section", "n_decided_discovery", "overlap_original",
                                 "overlap_tie_aware", "expected_chance"]].median().round(2).to_string())


if __name__ == "__main__":
    main()
