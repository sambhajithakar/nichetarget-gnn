"""Parity check: NumPy NicheTarget-GNN on the same simulated tissues as the PyTorch runs.

Re-runs the main model, LR-communication (niche-stratified) and DE on given
seeds with the NumPy backend and compares the AUPRC with the PyTorch results
stored in results/simulation/all_results_full-lr.json.

    python scripts/parity_numpy_vs_torch.py --seeds 1 2 --out results/parity_numpy
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nichetarget.simulate import simulate_tissue, CELL_TYPES  # noqa: E402
from nichetarget import np_graph as NG, np_model as NM, np_counterfactual as NC, np_baselines as NB  # noqa: E402

N_READOUT, K_GRAPH, N_NICHES = 15, 10, 5


def metrics(scores, genes, positives, k=10):
    y = np.array([g in positives for g in genes]).astype(int)
    r = NG.rankdata(-scores)
    order = np.lexsort((y, -scores))
    return {"auprc": NG.average_precision(y, scores), f"p@{k}": float(y[order[:k]].mean()),
            "ranks": {g: float(r[genes.index(g)]) for g in positives}}


def run(seed, epochs):
    t0 = time.time()
    sim = simulate_tissue(seed=seed, effect=1.0)
    X, genes = sim.X.astype(np.float32), sim.genes
    malignant = sim.cell_type == 0
    cells = np.where(malignant)[0]
    top = np.argsort(X[cells].var(0))[::-1][:50]
    z, _ = NG.pca(X[cells][:, top], 10)
    lab = NG.kmeans(z, 2, seed=seed)
    if sim.aggressiveness[cells][lab == 1].mean() < sim.aggressiveness[cells][lab == 0].mean():
        lab = 1 - lab
    target_cells, ref_cells = cells[lab == 1], cells[lab == 0]
    de = NB.de_score(X, target_cells, ref_cells)
    readout_idx = np.argsort(de)[::-1][:N_READOUT]
    readout = [genes[i] for i in readout_idx]
    programme = X[:, readout_idx].mean(1)
    g = NG.build_graph(X, sim.coords, sim.cell_type, len(CELL_TYPES), genes, sim.lr_pairs, readout, k=K_GRAPH)
    niche_hat = NG.kmeans(g.niche_comp, N_NICHES, seed=seed)
    sel = NC.selectivity(X, sim.cell_type, sim.X_normal, sim.normal_cell_type, len(CELL_TYPES))
    nbr = NB.neighbour_index(sim.coords, K_GRAPH)
    pos = sim.selective_targets
    out = {"seed": seed, "state_ARI": NG.adjusted_rand(sim.aggressiveness[cells] > 0.5, lab)}
    lr = NB.niche_stratified(NB.lr_communication, cells, niche_hat, X=X, nbr=nbr, malignant=malignant,
                             programme=programme, genes=genes, lr_pairs=sim.lr_pairs)
    out["LR-communication (niche-stratified)"] = metrics(lr, genes, pos)
    out["DE"] = metrics(de, genes, pos)
    model = NM.NicheTargetGNN(len(genes), len(CELL_TYPES), len(g.lr_idx), seed=seed)
    hist = NM.train_model(model, g, epochs=epochs, seed=seed)
    rows, _ = NC.scan_targets(model, g, genes, target_cells, ref_cells, niche_hat)
    eff = np.array([NC.niche_max(r[2]) for r in rows])
    out["NicheTarget-GNN"] = metrics(NC.combine_scores(eff, sel), genes, pos)
    out["final_train_loss"] = hist[-1]
    out["runtime_s"] = time.time() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--out", default="results/parity_numpy")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    here = os.path.join(os.path.dirname(__file__), "..")
    torch_res = {r["seed"]: r for r in json.load(open(os.path.join(here, "results/simulation/all_results_full-lr.json")))}
    res = []
    for s in a.seeds:
        r = run(s, a.epochs)
        t = torch_res.get(s, {}).get("metrics", {})
        r["torch"] = {m: t[m]["auprc"] for m in ["NicheTarget-GNN", "LR-communication (niche-stratified)", "DE"] if m in t}
        r["torch_final_train_loss"] = torch_res.get(s, {}).get("final_train_loss")
        print(json.dumps(r), flush=True)
        res.append(r)
        json.dump(res, open(os.path.join(a.out, "parity.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
