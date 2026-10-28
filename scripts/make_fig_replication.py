"""Figure 6: cross-patient replication (pre-specified) and read-out purity (post hoc).

    python scripts/make_fig_replication.py --summary results/replication/summary --out figures/Figure_6
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

LABELS = {"GNN": "NicheTarget-GNN", "LR-niche": "LR-comm. (niche)", "DE-niche": "DE (niche)", "DE": "DE"}
COL = {"GNN": "#0072B2", "LR-niche": "#E69F00", "DE-niche": "#009E73", "DE": "#999999"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ps = pd.read_csv(f"{a.summary}/replication_per_section.csv")
    pur = pd.read_csv(f"{a.summary}/readout_purity.csv")
    prim = ps[ps.passed_gate & ~ps.section.isin(["B1_2", "B1_3"])]
    plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.8), gridspec_kw={"width_ratios": [1.1, 1]})
    rng = np.random.default_rng(0)
    for i, m in enumerate(LABELS):
        v = prim[prim.method == m].vs_disc_top25_overlap.to_numpy()
        ax[0].scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=12, color=COL[m], alpha=0.8, lw=0)
        ax[0].plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color="k", lw=1.2)
    ax[0].axhline(25 * 25 / 265, color="k", ls=":", lw=0.8)
    ax[0].set_xticks(range(len(LABELS)), LABELS.values(), rotation=20, ha="right")
    ax[0].set_ylabel("Top-25 genes shared with Rep1")
    ax[0].set_ylim(-0.5, 25.5)
    ax[0].set_title("A  Replication across 17 donors", loc="left", fontsize=9, fontweight="bold")
    m = pur.merge(prim[prim.method == "GNN"][["section"]], on="section")
    ax[1].scatter(m.readout_purity, m.vs_disc_top25_overlap, s=16, color=COL["GNN"], lw=0)
    from scipy.stats import spearmanr
    rho, pv = spearmanr(m.readout_purity, m.vs_disc_top25_overlap)
    ax[1].text(0.03, 0.95, f"Spearman {rho:.2f}, p = {pv:.3f}", transform=ax[1].transAxes, va="top")
    ax[1].set_xlabel("Read-out purity")
    ax[1].set_ylabel("GNN top-25 genes shared with Rep1")
    ax[1].set_title("B  Read-out purity (post hoc)", loc="left", fontsize=9, fontweight="bold")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", dpi=300)


if __name__ == "__main__":
    main()
