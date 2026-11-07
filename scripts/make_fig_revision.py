"""Figure 7 (tie-aware, corrected 7 Nov 2026): pre-registered external validation on six GSE350571 tumours (results/revision).

    python scripts/make_fig_revision.py --root results/revision --out figures/Figure_7
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

LABELS = {"DE": "DE", "DE-niche": "DE (niche)", "GNN": "NicheTarget-GNN", "LR-niche": "LR-comm. (niche)"}
COL = {"GNN": "#0072B2", "LR-niche": "#E69F00", "DE-niche": "#009E73", "DE": "#555555"}
GNN = "NicheTarget-GNN (single model)"


def top(genes, scores, k=25):
    return set(np.asarray(genes)[np.argsort(-np.asarray(scores), kind="stable")[:k]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/revision")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    # tie-aware overlaps (scripts/audit_ties.py); DE under the contamination-aware rule is the unmasked DE
    parts = []
    for r in ["de", "malignant_specific"]:
        t = pd.read_csv(os.path.join(a.root, "tie_audit", f"external_{r}.csv"))
        if r == "malignant_specific":
            t = t[t.method != "DE"].replace({"method": {"DE-unmasked": "DE"}})
        parts.append(t.assign(rule=r))
    d = pd.concat(parts).rename(columns={"overlap_tie_aware": "vs_disc_top25_overlap"})
    plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [2.2, 1]})
    exp = (25 * 25 / d.n_common).median()
    for i, m in enumerate(LABELS):
        A = d.query("method == @m and rule == 'de'").set_index("section").vs_disc_top25_overlap
        B = d.query("method == @m and rule == 'malignant_specific'").set_index("section").vs_disc_top25_overlap
        s = sorted(set(A.index) & set(B.index))
        x0, x1 = i - 0.18, i + 0.18
        for t in s:
            ax[0].plot([x0, x1], [A[t], B[t]], color=COL[m], alpha=0.45, lw=0.8)
        ax[0].scatter([x0] * len(s), A[s], s=12, facecolor="white", edgecolor=COL[m], lw=0.8, zorder=3)
        ax[0].scatter([x1] * len(s), B[s], s=12, color=COL[m], lw=0, zorder=3)
        ax[0].plot([x0 - 0.1, x0 + 0.1], [np.median(A[s])] * 2, color="k", lw=1.2)
        ax[0].plot([x1 - 0.1, x1 + 0.1], [np.median(B[s])] * 2, color="k", lw=1.2)
    ax[0].axhline(exp, color="k", ls=":", lw=0.8)
    ax[0].set_xticks(range(len(LABELS)), LABELS.values(), rotation=15, ha="right")
    ax[0].set_ylabel("Top-25 targets shared with discovery")
    ax[0].set_ylim(-0.5, 25.5)
    ax[0].scatter([], [], s=12, facecolor="white", edgecolor="k", label="default read-out")
    ax[0].scatter([], [], s=12, color="k", label="contamination-aware read-out")
    ax[0].legend(frameon=False, loc="upper left", fontsize=7)
    ax[0].set_title("A  Six unseen tumours (GSE350571)", loc="left", fontsize=9, fontweight="bold")

    # B: GNN within-section seed overlap vs between-tumour overlap
    rows = []
    for r, lab in [("de", "default"), ("malignant_specific", "contam.-aware")]:
        tops = {}
        for s in [1, 2, 3]:
            g = pd.read_csv(os.path.join(a.root, "rep1_g180", f"{r}_s{s}", "gene_scores.csv"))
            g = g[~g.is_readout]
            tops[s] = top(g.gene, g[GNN])
        seeds = [len(tops[i] & tops[j]) for i, j in [(1, 2), (1, 3), (2, 3)]]
        betw = d.query("method == 'GNN' and rule == @r").vs_disc_top25_overlap.to_numpy()
        rows.append((lab, seeds, betw))
    for k, (lab, seeds, betw) in enumerate(rows):
        ax[1].scatter([k * 2 + 0.0] * 3, seeds, s=14, color=COL["GNN"], lw=0)
        ax[1].scatter([k * 2 + 0.8] * len(betw), betw, s=14, facecolor="white", edgecolor=COL["GNN"], lw=0.8)
    ax[1].axhline(exp, color="k", ls=":", lw=0.8)
    ax[1].set_xticks([0, 0.8, 2, 2.8], ["seeds", "tumours", "seeds", "tumours"])
    for k, (lab, _, _) in enumerate(rows):
        ax[1].text(k * 2 + 0.4, -6.5, lab, ha="center", fontsize=7)
    ax[1].set_ylim(-0.5, 25.5)
    ax[1].set_ylabel("GNN top-25 overlap")
    ax[1].set_title("B  GNN: seeds vs tumours", loc="left", fontsize=9, fontweight="bold")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", dpi=300)


if __name__ == "__main__":
    main()
