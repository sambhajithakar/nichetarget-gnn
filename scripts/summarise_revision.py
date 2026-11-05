"""Endpoints of results/revision/PRESPECIFIED_REVISION.md (external cohort GSE350571, 180 shared genes).

Per-tumour metrics come from summarise_replication.py (run once per read-out rule, discovery = Rep1-G180 with
the same rule, seed 1) and read-out purity from readout_purity.purity. This script adds the across-tumour tests:
H1 sign test, H2 paired Wilcoxon (contamination-aware vs default rule), H3 purity vs reproducibility,
H4 seed stability of the discovery ranking.

    python scripts/summarise_revision.py --root results/revision --old-purity <17-donor readout_purity.csv> \
        --out results/revision/summary
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

sys.path.insert(0, os.path.dirname(__file__))
from readout_purity import purity  # noqa: E402
from summarise_replication import hypergeom_free_sign, top, TOPK  # noqa: E402

RULES = ["malignant_specific", "de"]
GNN = "NicheTarget-GNN (single model)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/revision")
    ap.add_argument("--old-purity", required=True, help="readout_purity.csv of the 17-donor analysis (default rule)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    lines = []

    per = {}
    for r in RULES:
        d = pd.read_csv(os.path.join(a.root, "summary", r, "replication_per_section.csv"))
        d["rule"] = r
        d["expected_overlap"] = TOPK * TOPK / d.n_common
        d["readout_purity"] = [purity(os.path.join(a.root, "external", r, s))[0] for s in d.section]
        per[r] = d
    allp = pd.concat(per.values())
    allp.to_csv(os.path.join(a.out, "external_per_tumour.csv"), index=False)

    # H1 + baselines: sign tests per rule and method
    for r in RULES:
        for m, g in per[r][per[r].passed_gate].groupby("method"):
            k = int((g.vs_disc_top25_overlap > g.expected_overlap).sum())
            ks = int((g.vs_disc_spearman > 0).sum())
            lines.append(f"[{r}] {m}: n={len(g)} overlap median={g.vs_disc_top25_overlap.median():.1f} "
                         f"(expected {g.expected_overlap.median():.1f}); above expectation {k}/{len(g)} "
                         f"sign p={hypergeom_free_sign(k, len(g)):.3f}; spearman median={g.vs_disc_spearman.median():.3f}"
                         f", >0 in {ks}/{len(g)} p={hypergeom_free_sign(ks, len(g)):.3f}")
    # H2: paired, contamination-aware minus default
    for m in sorted(per["de"].method.unique()):
        A = per["malignant_specific"].query("method == @m and passed_gate").set_index("section")
        B = per["de"].query("method == @m and passed_gate").set_index("section")
        s = sorted(set(A.index) & set(B.index))
        for col in ["vs_disc_top25_overlap", "vs_disc_spearman"]:
            diff = A.loc[s, col].to_numpy() - B.loc[s, col].to_numpy()
            p = wilcoxon(diff, alternative="greater").pvalue if np.any(diff != 0) else 1.0
            lines.append(f"H2 {m} {col}: n={len(s)} median diff={np.median(diff):.3f} "
                         f"improved in {(diff > 0).sum()}/{len(s)}; Wilcoxon one-sided p={p:.3f}")
    # H3: purity (default rule) vs GNN overlap, new tumours alone and pooled with the 17 donors
    g = per["de"].query("method == 'GNN' and passed_gate")
    rho, p = spearmanr(g.readout_purity, g.vs_disc_top25_overlap)
    lines.append(f"H3 new tumours: n={len(g)} spearman(purity, GNN overlap)={rho:.3f} p={p:.3f}")
    old = pd.read_csv(a.old_purity).dropna(subset=["vs_disc_top25_overlap"])
    old = old[old.passed_gate.astype(str) == "True"]
    pur = np.r_[old.readout_purity, g.readout_purity]; ov = np.r_[old.vs_disc_top25_overlap, g.vs_disc_top25_overlap]
    rho, p = spearmanr(pur, ov)
    lines.append(f"H3 pooled (17 + new): n={len(pur)} spearman={rho:.3f} p={p:.4f}")
    for lab, msk in [("gated purity>=0.5", g.readout_purity >= 0.5), ("flagged purity<0.5", g.readout_purity < 0.5)]:
        lines.append(f"H3 new tumours {lab}: n={int(msk.sum())} median GNN overlap="
                     f"{g.vs_disc_top25_overlap[msk].median() if msk.any() else float('nan')}")
    # H4: seed stability of discovery Rep1-G180
    for r in RULES:
        tops = {}
        for s in [1, 2, 3]:
            p_ = os.path.join(a.root, "rep1_g180", f"{r}_s{s}", "gene_scores.csv")
            if os.path.exists(p_):
                d = pd.read_csv(p_)
                d = d[~d.is_readout]
                tops[s] = top(d.gene.to_numpy(), d[GNN].to_numpy())
        pairs = [(i, j) for i in tops for j in tops if i < j]
        jac = [len(tops[i] & tops[j]) / len(tops[i] | tops[j]) for i, j in pairs]
        ov = [len(tops[i] & tops[j]) for i, j in pairs]
        lines.append(f"H4 [{r}] seeds {sorted(tops)}: top-25 overlap {ov}, Jaccard {np.round(jac, 3).tolist()}")
    txt = "\n".join(lines)
    open(os.path.join(a.out, "revision_endpoints.txt"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
