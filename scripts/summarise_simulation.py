"""Aggregate simulation results into tables and figures for the manuscript.

Usage:
    python scripts/summarise_simulation.py results/simulation figures_out_dir
"""
from __future__ import annotations

import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

GENES = ["L1", "R1", "L2", "R2", "D1"]


def load(res_dir):
    rows, meta = [], []
    for f in sorted(glob.glob(os.path.join(res_dir, "all_results_*.json"))):
        scen = os.path.basename(f).replace("all_results_", "").replace(".json", "")
        for r in json.load(open(f)):
            meta.append({"scenario": scen, "seed": r["seed"], "state_ARI": r["state_ARI"],
                         "niche_ARI": r["niche_ARI"], "readout_overlap": r["readout_overlap"],
                         "readout_R2": r.get("readout_R2", np.nan), "runtime_s": r["runtime_s"],
                         **{f"nicheacc::{k}": v for k, v in r["niche_localisation_accuracy"].items()}})
            for m, v in r["metrics"].items():
                rows.append({"scenario": scen, "seed": r["seed"], "method": m, "AUPRC": v["auprc"],
                             "P@10": v["p@10"], "R@10": v["recall@10"],
                             **{f"rank_{g}": v["ranks"][g] for g in GENES}})
    return pd.DataFrame(rows), pd.DataFrame(meta)


def table(df):
    agg = df.groupby(["scenario", "method"]).agg(
        AUPRC_mean=("AUPRC", "mean"), AUPRC_sd=("AUPRC", "std"),
        P10_mean=("P@10", "mean"), R10_mean=("R@10", "mean"),
        **{f"median_rank_{g}": (f"rank_{g}", "median") for g in GENES}).reset_index()
    return agg.sort_values(["scenario", "AUPRC_mean"], ascending=[True, False])


MAIN = ["DE", "DE (niche-stratified)", "NeighbourhoodCorr", "NeighbourhoodCorr (niche-stratified)",
        "LR-communication", "LR-communication (niche-stratified)",
        "LR-communication (niche-stratified)+Selectivity", "NicheTarget-GNN",
        "NicheTarget-GNN (ensemble of 3)", "Hybrid (GNN ensemble + LR-communication)"]


def fig_main(df, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    for ax, scen, title in zip(axes, ["full-lr", "hidden-L1-R1"],
                               ["A  Complete LR database", "B  L1–R1 missing from LR database"]):
        d = df[(df.scenario == scen) & (df.method.isin(MAIN))]
        if d.empty:
            ax.set_visible(False)
            continue
        order = [m for m in MAIN if m in d.method.unique()]
        means = [d[d.method == m].AUPRC.mean() for m in order]
        sds = [d[d.method == m].AUPRC.std() for m in order]
        colors = ["#2a6fdb" if m.startswith(("NicheTarget-GNN", "Hybrid")) else "#9aa5b1" for m in order]
        y = np.arange(len(order))
        ax.barh(y, means, xerr=sds, color=colors, capsize=3)
        for m_i, m in enumerate(order):
            pts = d[d.method == m].AUPRC.values
            ax.scatter(pts, np.full(len(pts), m_i), s=10, color="black", zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(order, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("AUPRC (selective targets)")
        ax.set_title(title, loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def fig_ablation(df, out):
    d = df[df.scenario == "full-lr"]
    order = ["NicheTarget-GNN", "NicheTarget-GNN (no selectivity)", "NicheTarget-GNN (global efficacy)",
             "w/o spatial message passing", "w/o niche conditioning", "w/o LR edge features"]
    order = [m for m in order if m in d.method.unique()]
    if not order:
        return
    fig, ax = plt.subplots(figsize=(6, 3.4))
    means = [d[d.method == m].AUPRC.mean() for m in order]
    sds = [d[d.method == m].AUPRC.std() for m in order]
    y = np.arange(len(order))
    ax.barh(y, means, xerr=sds, color=["#2a6fdb"] + ["#9aa5b1"] * (len(order) - 1), capsize=3)
    for m_i, m in enumerate(order):
        pts = d[d.method == m].AUPRC.values
        ax.scatter(pts, np.full(len(pts), m_i), s=10, color="black", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("AUPRC (selective targets)")
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def fig_ranks(df, out):
    d = df[(df.scenario == "full-lr") & (df.method.isin(["DE", "NeighbourhoodCorr", "LR-communication (niche-stratified)", "NicheTarget-GNN"]))]
    if d.empty:
        return
    methods = ["DE", "NeighbourhoodCorr", "LR-communication (niche-stratified)", "NicheTarget-GNN"]
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    w = 0.2
    for i, m in enumerate(methods):
        med = [d[d.method == m][f"rank_{g}"].median() for g in GENES]
        ax.bar(np.arange(len(GENES)) + (i - 1.5) * w, med, width=w, label=m,
               color=["#c9d1d9", "#9aa5b1", "#5b6770", "#2a6fdb"][i])
    ax.set_xticks(np.arange(len(GENES)))
    ax.set_xticklabels(GENES)
    ax.set_yscale("log")
    ax.set_ylabel("Median rank (of 200 genes, log)")
    ax.invert_yaxis()
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def main(res_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    df, meta = load(res_dir)
    df.to_csv(os.path.join(out_dir, "simulation_metrics_long.csv"), index=False)
    t = table(df)
    t.to_csv(os.path.join(out_dir, "simulation_summary.csv"), index=False)
    meta.to_csv(os.path.join(out_dir, "simulation_meta.csv"), index=False)
    fig_main(df, os.path.join(out_dir, "Fig3_benchmark.png"))
    fig_ablation(df, os.path.join(out_dir, "Fig4_ablation.png"))
    fig_ranks(df, os.path.join(out_dir, "Fig5_target_ranks.png"))
    pd.set_option("display.width", 250)
    print(t.to_string(index=False))
    print(meta.groupby("scenario").mean(numeric_only=True).to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
