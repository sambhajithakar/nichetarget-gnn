"""Simulation benchmark: can each method recover planted, selective, niche-specific targets?

Usage:
    python scripts/run_simulation.py --seeds 0 1 2 3 4 --out results/simulation
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, average_precision_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nichetarget.simulate import simulate_tissue, CELL_TYPES  # noqa: E402
from nichetarget.graph import build_graph  # noqa: E402
from nichetarget.model import NicheTargetGNN, train_model  # noqa: E402
from nichetarget.counterfactual import scan_targets, selectivity, combine_scores, niche_max  # noqa: E402
from nichetarget.model import make_input  # noqa: E402
from nichetarget import baselines as B  # noqa: E402

N_READOUT = 15
K_GRAPH = 10
N_NICHES = 5


def annotate_states(X, malignant, seed):
    """Unsupervised two-state annotation of malignant cells (k-means on PCA)."""
    cells = np.where(malignant)[0]
    Xm = X[cells]
    var = Xm.var(0)
    top = np.argsort(var)[::-1][:50]
    z = PCA(n_components=10, random_state=seed).fit_transform(Xm[:, top])
    lab = KMeans(n_clusters=2, n_init=10, random_state=seed).fit_predict(z)
    return cells, lab


def metrics(scores: np.ndarray, genes, positives, k=10):
    """Tie-aware metrics.

    Ranks are average ranks over tied scores (e.g., the many genes that a
    baseline scores as exactly zero). Top-k precision/recall break ties
    pessimistically (positives placed after negatives with the same score),
    so a method cannot gain from arbitrary tie ordering.
    """
    from scipy.stats import rankdata
    y = np.array([g in positives for g in genes]).astype(int)
    r = rankdata(-scores, method="average")
    ranks = {g: float(r[genes.index(g)]) for g in positives}
    order = np.lexsort((y, -scores))       # primary: score desc; ties: negatives first
    return {
        "auprc": float(average_precision_score(y, scores)),
        f"p@{k}": float(y[order[:k]].mean()),
        f"recall@{k}": float(y[order[:k]].sum() / y.sum()),
        "ranks": ranks,
    }


def infer_niches(niche_comp, seed):
    """Data-driven niches: k-means on neighbourhood cell-type composition."""
    return KMeans(n_clusters=N_NICHES, n_init=10, random_state=seed).fit_predict(niche_comp)


def map_niches(inferred, true):
    """Map each inferred niche to the true niche it overlaps most (for evaluation only)."""
    return {int(n): int(np.bincount(true[inferred == n]).argmax()) for n in np.unique(inferred)}


def run_seed(seed, epochs, out_dir, effect, hide_lr=(), main_only=False, n_ensemble=1):
    t0 = time.time()
    sim = simulate_tissue(seed=seed, effect=effect)
    X, genes = sim.X, sim.genes
    n_types = len(CELL_TYPES)
    malignant = sim.cell_type == 0

    # state annotation (target = more aggressive cluster, as a pathologist would name it)
    cells, lab = annotate_states(X, malignant, seed)
    agg_true = sim.aggressiveness[cells] > 0.5
    if sim.aggressiveness[cells][lab == 1].mean() < sim.aggressiveness[cells][lab == 0].mean():
        lab = 1 - lab
    ari = adjusted_rand_score(agg_true, lab)
    target_cells, ref_cells = cells[lab == 1], cells[lab == 0]

    de = B.de_score(X, target_cells, ref_cells)
    readout_idx = np.argsort(de)[::-1][:N_READOUT]
    readout = [genes[i] for i in readout_idx]
    readout_overlap = len(set(readout) & set(sim.readout_genes)) / N_READOUT
    programme = X[:, readout_idx].mean(1)

    lr_known = [p for p in sim.lr_pairs if p[0] not in hide_lr and p[1] not in hide_lr]
    g = build_graph(X, sim.coords, sim.cell_type, n_types, genes, lr_known, readout, k=K_GRAPH)
    niche_hat = infer_niches(g.niche_comp.numpy(), seed)
    nmap = map_niches(niche_hat, sim.niche)
    sel = selectivity(X, sim.cell_type, sim.X_normal, sim.normal_cell_type, n_types)

    positives = sim.selective_targets
    res = {"seed": seed, "effect": effect, "hidden_lr": list(hide_lr), "state_ARI": ari,
           "readout_overlap": readout_overlap, "niche_ARI": adjusted_rand_score(sim.niche, niche_hat)}
    table = {}

    # ---- baselines --------------------------------------------------------------
    mal_cells = np.where(malignant)[0]
    base_scores = {
        "DE": de,
        "DE (niche-stratified)": B.de_niche_max(X, target_cells, ref_cells, niche_hat),
        "NeighbourhoodCorr": B.neighbourhood_correlation(X, sim.coords, malignant, programme, k=K_GRAPH),
        "NeighbourhoodCorr (niche-stratified)": B.niche_stratified(
            B.neighbourhood_correlation, mal_cells, niche_hat, X=X, coords=sim.coords,
            malignant=malignant, programme=programme, k=K_GRAPH),
        "LR-communication": B.lr_communication(X, sim.coords, malignant, programme, genes, lr_known, k=K_GRAPH),
        "LR-communication (niche-stratified)": B.niche_stratified(
            B.lr_communication, mal_cells, niche_hat, X=X, coords=sim.coords, malignant=malignant,
            programme=programme, genes=genes, lr_pairs=lr_known, k=K_GRAPH),
    }
    for name, s in base_scores.items():
        table[name] = metrics(s, genes, positives)
        s_pos = np.clip(s, 0, None)
        table[name + "+Selectivity"] = metrics(s_pos / (s_pos.max() + 1e-12) * sel, genes, positives)

    # ---- NicheTarget-GNN and ablations --------------------------------------------
    variants = {
        "NicheTarget-GNN": dict(),
        "w/o spatial message passing": dict(use_spatial=False),
        "w/o niche conditioning": dict(use_niche=False),
        "w/o LR edge features": dict(use_lr=False),
    }
    if main_only:
        variants = {"NicheTarget-GNN": dict()}
    niche_acc = {}
    per_gene_full = None
    for name, kw in variants.items():
        torch.manual_seed(seed)
        model = NicheTargetGNN(len(genes), n_types, g.lr_idx.shape[0], **kw)
        hist = train_model(model, g, epochs=epochs, seed=seed)
        rows = scan_targets(model, g, genes, target_cells, ref_cells, niche_hat)
        eff_global = np.array([r[1] for r in rows])
        eff = np.array([niche_max(r[2]) for r in rows])
        if name == "NicheTarget-GNN":
            table[name + " (global efficacy)"] = metrics(combine_scores(eff_global, sel), genes, positives)
            with torch.no_grad():
                pred = model(make_input(g.x, g.readout_idx), g).numpy()
            yt, yp = X[target_cells.tolist() + ref_cells.tolist()][:, readout_idx], pred[target_cells.tolist() + ref_cells.tolist()][:, readout_idx]
            res["readout_R2"] = float(1 - ((yt - yp) ** 2).sum() / ((yt - yt.mean(0)) ** 2).sum())
        table[name + " (no selectivity)"] = metrics(combine_scores(eff, None), genes, positives)
        table[name] = metrics(combine_scores(eff, sel), genes, positives)
        # niche localisation for the planted niche-specific targets
        ok = []
        for gene, true_n in sim.niche_specific.items():
            per = rows[genes.index(gene)][2]
            if per:
                ok.append(nmap[max(per, key=per.get)] == true_n)
        niche_acc[name] = float(np.mean(ok)) if ok else float("nan")
        if name == "NicheTarget-GNN":
            per_gene_full = pd.DataFrame({
                "gene": genes, "efficacy": eff, "selectivity": sel,
                "score": combine_scores(eff, sel),
                "efficacy_global": eff_global,
                "best_niche_true_label": [nmap[max(r[2], key=r[2].get)] if r[2] else -1 for r in rows],
                **{f"efficacy_niche{n}": [r[2].get(n, np.nan) for r in rows] for n in range(N_NICHES)},
            }).sort_values("score", ascending=False)
            res["final_train_loss"] = hist[-1]

    # ---- v2: ensemble of independently initialised models, and a hybrid with LR-communication
    if n_ensemble > 1:
        from scipy.stats import rankdata
        effs = []
        per_niche_sum = {}
        for e in range(n_ensemble):
            s_e = seed * 100 + e
            torch.manual_seed(s_e)
            m_e = NicheTargetGNN(len(genes), n_types, g.lr_idx.shape[0])
            train_model(m_e, g, epochs=epochs, seed=s_e)
            rows_e = scan_targets(m_e, g, genes, target_cells, ref_cells, niche_hat)
            for j, r in enumerate(rows_e):
                for n, v in r[2].items():
                    per_niche_sum[(j, n)] = per_niche_sum.get((j, n), 0.0) + v / n_ensemble
        eff_ens = np.zeros(len(genes))
        for j in range(len(genes)):
            vals = [v for (jj, n), v in per_niche_sum.items() if jj == j]
            eff_ens[j] = max(vals) if vals else 0.0
        gnn_ens = combine_scores(eff_ens, sel)
        table[f"NicheTarget-GNN (ensemble of {n_ensemble})"] = metrics(gnn_ens, genes, positives)
        lr_s = base_scores["LR-communication (niche-stratified)"]
        G_ = len(genes)
        hybrid = 0.5 * rankdata(gnn_ens) / G_ + 0.5 * rankdata(lr_s) / G_
        table["Hybrid (GNN ensemble + LR-communication)"] = metrics(hybrid, genes, positives)

    res["metrics"] = table
    res["niche_localisation_accuracy"] = niche_acc
    res["runtime_s"] = time.time() - t0
    os.makedirs(out_dir, exist_ok=True)
    tag = "hidden-" + "-".join(hide_lr) if hide_lr else "full-lr"
    per_gene_full.to_csv(os.path.join(out_dir, f"ranking_seed{seed}_{tag}.csv"), index=False)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--effects", type=float, nargs="+", default=[1.0])
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--out", default="results/simulation")
    ap.add_argument("--n-ensemble", type=int, default=1, help="v2: number of models averaged")
    ap.add_argument("--main-only", action="store_true", help="train only the full model (skip ablations)")
    ap.add_argument("--hide-lr", nargs="*", default=[], help="genes whose LR pairs are removed from the curated list")
    args = ap.parse_args()
    torch.set_num_threads(4)
    all_res = []
    for eff in args.effects:
        for s in args.seeds:
            r = run_seed(s, args.epochs, args.out, eff, tuple(args.hide_lr), args.main_only, args.n_ensemble)
            print(json.dumps({k: v for k, v in r.items() if k != "metrics"}), flush=True)
            for m, v in r["metrics"].items():
                print(f"  {m:45s} AUPRC={v['auprc']:.3f} P@10={v['p@10']:.2f} ranks={v['ranks']}", flush=True)
            all_res.append(r)
            tag = "hidden-" + "-".join(args.hide_lr) if args.hide_lr else "full-lr"
            with open(os.path.join(args.out, f"all_results_{tag}.json"), "w") as f:
                json.dump(all_res, f, indent=1)


if __name__ == "__main__":
    main()
