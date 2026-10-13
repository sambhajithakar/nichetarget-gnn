"""Published spatial ligand-receptor (LR) tools on the simulation benchmark (seeds 1-5, both scenarios).

Tools (the published code itself, not re-implementations):
  * COMMOT (Cang et al., Nat Methods 2023) -- collective optimal transport,
    ``commot.tl.spatial_communication``; installed from github.com/zcang/COMMOT.
  * LIANA+ (Dimitrov et al., Nat Cell Biol 2024) -- spatially informed bivariate
    metrics, ``liana.mt.bivariate``; installed from bioconda.

Both tools receive exactly the curated LR list that the in-house LR-communication
baseline receives (all 8 pairs in the complete-database scenario; L1-R1 removed
in the incomplete-database scenario). NicheNet's prior model is human-gene based
and cannot be applied to the synthetic genes of the simulator.

Pre-specified conversion of tool output to a gene ranking (same rule as the
in-house LR-communication baseline, nichetarget/baselines.py::lr_communication):
  per-cell tool score for an LR pair (COMMOT: signal received by the cell through
  the pair, obsm['commot-<db>-sum-receiver']['r-L-R']; LIANA+: local score of the
  pair at the cell, default local metric 'cosine'; local Moran's R added post hoc
  as a sensitivity analysis, see run_seed) -> Spearman correlation with
  the state programme score across malignant cells -> both genes of the pair get
  the maximum pair score; genes outside the database get 0. Niche-stratified
  variant: the same within each inferred niche (>= 30 malignant cells), maximum
  over niches. "+Selectivity" variants use the same selectivity score as the
  in-house baselines.
A secondary, state-agnostic "tool-native" ranking is also reported: COMMOT =
total received signal of the pair summed over malignant cells; LIANA+ = global
bivariate Moran's R of the pair (all cells); both genes inherit the pair maximum.

Spatial scale (pre-specified, no tuning): r = mean distance to the 10th nearest
neighbour (k = 10, the in-house graph). COMMOT dis_thr = r; LIANA+
spatial_neighbors(bandwidth = r, kernel = 'gaussian', cutoff = 0.1, set_diag = True).
All other tool parameters are defaults.

State annotation, read-out, niches and metrics are computed by verbatim copies of
the functions in scripts/run_simulation.py (that script imports torch, which is
not needed here). The in-house DE and LR-communication baselines are recomputed
and compared with results/simulation/all_results_*.json as a reproduction check.

    python scripts/run_published_comparators_simulation.py --seeds 1 2 3 4 5 --out results/published_comparators/simulation
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import rankdata, spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, average_precision_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nichetarget.simulate import simulate_tissue, CELL_TYPES  # noqa: E402
from nichetarget import baselines as B  # noqa: E402
from nichetarget.np_counterfactual import selectivity  # noqa: E402  (identical formula to counterfactual.selectivity)

warnings.filterwarnings("ignore")
N_READOUT, K_GRAPH, N_NICHES, MIN_CELLS = 15, 10, 5, 30
GENES5 = ["L1", "R1", "L2", "R2", "D1"]


# ---- verbatim copies from scripts/run_simulation.py / nichetarget/graph.py ------------------
def annotate_states(X, malignant, seed):
    cells = np.where(malignant)[0]
    Xm = X[cells]
    var = Xm.var(0)
    top = np.argsort(var)[::-1][:50]
    z = PCA(n_components=10, random_state=seed).fit_transform(Xm[:, top])
    lab = KMeans(n_clusters=2, n_init=10, random_state=seed).fit_predict(z)
    return cells, lab


def metrics(scores, genes, positives, k=10):
    y = np.array([g in positives for g in genes]).astype(int)
    r = rankdata(-scores, method="average")
    ranks = {g: float(r[genes.index(g)]) for g in positives}
    order = np.lexsort((y, -scores))
    return {"auprc": float(average_precision_score(y, scores)), f"p@{k}": float(y[order[:k]].mean()),
            f"recall@{k}": float(y[order[:k]].sum() / y.sum()), "ranks": ranks}


def niche_composition(coords, cell_type, n_types, k=15):
    tree = cKDTree(coords)
    _, nbr = tree.query(coords, k=k + 1)
    onehot = np.eye(n_types)[cell_type]
    return onehot[nbr].mean(1)


def infer_niches(niche_comp, seed):
    return KMeans(n_clusters=N_NICHES, n_init=10, random_state=seed).fit_predict(niche_comp)
# ------------------------------------------------------------------------------------------------


def spatial_scale(coords, k=K_GRAPH):
    d, _ = cKDTree(coords).query(coords, k=k + 1)
    return float(d[:, k].mean())


def pair_to_gene(pair_scores: dict, genes):
    """Both genes of a pair receive the maximum score of the pairs they belong to; others 0."""
    gi = {g: i for i, g in enumerate(genes)}
    out = np.zeros(len(genes))
    for (l, r), c in pair_scores.items():
        out[gi[l]] = max(out[gi[l]], c)
        out[gi[r]] = max(out[gi[r]], c)
    return out


def corr_rule(per_cell: dict, programme, cells):
    """Pair score = Spearman(per-cell tool score, programme) across the given malignant cells."""
    out = {}
    for p, v in per_cell.items():
        x = v[cells]
        if x.std() < 1e-8:
            continue
        c = spearmanr(x, programme[cells]).correlation
        out[p] = 0.0 if np.isnan(c) else float(c)
    return out


def gene_scores_from_tool(per_cell, programme, genes, mal_cells, niche):
    glob = pair_to_gene(corr_rule(per_cell, programme, mal_cells), genes)
    best = None
    for n in np.unique(niche[mal_cells]):
        cells = mal_cells[niche[mal_cells] == n]
        if len(cells) < MIN_CELLS:
            continue
        s = pair_to_gene(corr_rule(per_cell, programme, cells), genes)
        best = s if best is None else np.maximum(best, s)
    return glob, best


def run_commot(X, coords, genes, lr_pairs, r):
    import anndata
    import commot as ct
    from scipy import sparse
    ad = anndata.AnnData(X=np.asarray(X, np.float64), var=pd.DataFrame(index=genes))
    ad.obsm["spatial"] = np.asarray(coords, float)
    # sparse distance matrix within r (COMMOT accepts a sparse obsp['spatial_distance'])
    tree = cKDTree(ad.obsm["spatial"])
    D = tree.sparse_distance_matrix(tree, r, output_type="coo_matrix").tocsr()
    ad.obsp["spatial_distance"] = D
    df = pd.DataFrame([[l, rr, f"{l}-{rr}"] for l, rr in lr_pairs])
    ct.tl.spatial_communication(ad, database_name="db", df_ligrec=df, dis_thr=r, heteromeric=False,
                                pathway_sum=False)
    rec = ad.obsm["commot-db-sum-receiver"]
    per_cell = {(l, rr): rec[f"r-{l}-{rr}"].to_numpy(float) for l, rr in lr_pairs if f"r-{l}-{rr}" in rec}
    return per_cell


def run_liana(X, coords, genes, lr_pairs, r, local_name="cosine"):
    import anndata
    import liana as li
    from scipy import sparse
    ad = anndata.AnnData(X=sparse.csr_matrix(np.asarray(X, np.float32)), var=pd.DataFrame(index=genes))
    ad.obsm["spatial"] = np.asarray(coords, float)
    li.ut.spatial_neighbors(ad, bandwidth=r, cutoff=0.1, kernel="gaussian", set_diag=True)
    res = pd.DataFrame(lr_pairs, columns=["ligand", "receptor"])
    lr = li.mt.bivariate(ad, resource=res, local_name=local_name, global_name="morans", n_perms=None,
                         use_raw=False, verbose=False)
    Xl = lr.X.toarray() if hasattr(lr.X, "toarray") else np.asarray(lr.X)
    per_cell, glob = {}, {}
    for j, (inter, row) in enumerate(lr.var.iterrows()):
        p = (row["ligand"], row["receptor"])
        per_cell[p] = Xl[:, j]
        glob[p] = float(row["morans"])
    return per_cell, glob, list(lr.var.columns)


def run_seed(seed, hide_lr, ref_json):
    t0 = time.time()
    sim = simulate_tissue(seed=seed, effect=1.0)
    X, genes = sim.X, sim.genes
    malignant = sim.cell_type == 0
    cells, lab = annotate_states(X, malignant, seed)
    if sim.aggressiveness[cells][lab == 1].mean() < sim.aggressiveness[cells][lab == 0].mean():
        lab = 1 - lab
    target_cells, ref_cells = cells[lab == 1], cells[lab == 0]
    de = B.de_score(X, target_cells, ref_cells)
    readout_idx = np.argsort(de)[::-1][:N_READOUT]
    programme = X[:, readout_idx].mean(1)
    lr_known = [p for p in sim.lr_pairs if p[0] not in hide_lr and p[1] not in hide_lr]
    niche_hat = infer_niches(niche_composition(sim.coords, sim.cell_type, len(CELL_TYPES), 15), seed)
    sel = selectivity(X, sim.cell_type, sim.X_normal, sim.normal_cell_type, len(CELL_TYPES))
    mal_cells = np.where(malignant)[0]
    positives = sim.selective_targets
    r = spatial_scale(sim.coords)

    scores = {}
    # reproduction check of the in-house baselines
    scores["[check] DE"] = de
    scores["[check] LR-communication"] = B.lr_communication(X, sim.coords, malignant, programme, genes, lr_known,
                                                            k=K_GRAPH)
    scores["[check] LR-communication (niche-stratified)"] = B.niche_stratified(
        B.lr_communication, mal_cells, niche_hat, X=X, coords=sim.coords, malignant=malignant,
        programme=programme, genes=genes, lr_pairs=lr_known, k=K_GRAPH)

    tc = time.time()
    pc_commot = run_commot(X, sim.coords, genes, lr_known, r)
    t_commot = time.time() - tc
    g_, n_ = gene_scores_from_tool(pc_commot, programme, genes, mal_cells, niche_hat)
    scores["COMMOT"] = g_
    scores["COMMOT (niche-stratified)"] = n_
    scores["COMMOT native (total received signal, malignant)"] = pair_to_gene(
        {p: float(v[mal_cells].sum()) for p, v in pc_commot.items()}, genes)

    tl = time.time()
    pc_liana, glob_liana, liana_cols = run_liana(X, sim.coords, genes, lr_known, r)
    t_liana = time.time() - tl
    g_, n_ = gene_scores_from_tool(pc_liana, programme, genes, mal_cells, niche_hat)
    scores["LIANA+ bivariate"] = g_
    scores["LIANA+ bivariate (niche-stratified)"] = n_
    scores["LIANA+ native (global Moran's R)"] = pair_to_gene(
        {p: (0.0 if np.isnan(v) else v) for p, v in glob_liana.items()}, genes)
    # sensitivity analysis added AFTER the default (cosine) result was seen on seed 1: local bivariate
    # Moran's R (x_i * (W y)_i + y_i * (W x)_i), which, unlike cosine, links a cell to its neighbours' expression
    pc_lm, _, _ = run_liana(X, sim.coords, genes, lr_known, r, local_name="morans")
    g_, n_ = gene_scores_from_tool(pc_lm, programme, genes, mal_cells, niche_hat)
    scores["LIANA+ bivariate, local Moran's R [post hoc]"] = g_
    scores["LIANA+ bivariate, local Moran's R (niche-stratified) [post hoc]"] = n_

    table = {}
    for name, s in scores.items():
        table[name] = metrics(s, genes, positives)
        s_pos = np.clip(s, 0, None)
        table[name + "+Selectivity"] = metrics(s_pos / (s_pos.max() + 1e-12) * sel, genes, positives)

    # reproduction check against the stored in-house results
    check = {}
    ref = [x for x in ref_json if x["seed"] == seed]
    if ref:
        ref = ref[0]["metrics"]
        for nm in ["DE", "LR-communication", "LR-communication (niche-stratified)"]:
            a, b = table[f"[check] {nm}"], ref[nm]
            check[nm] = {"auprc_now": a["auprc"], "auprc_stored": b["auprc"],
                         "ranks_equal": a["ranks"] == b["ranks"],
                         "match": abs(a["auprc"] - b["auprc"]) < 1e-9 and a["ranks"] == b["ranks"]}
    per_gene = pd.DataFrame({"gene": genes, **{k: v for k, v in scores.items() if not k.startswith("[check]")}})
    return {"seed": seed, "hidden_lr": list(hide_lr), "lr_pairs_given": [list(p) for p in lr_known],
            "spatial_scale_r": r, "commot_pairs_scored": len(pc_commot), "liana_pairs_scored": len(pc_liana),
            "liana_var_columns": liana_cols, "runtime_commot_s": t_commot, "runtime_liana_s": t_liana,
            "runtime_s": time.time() - t0, "reproduction_check": check, "metrics": table}, per_gene


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--out", default="results/published_comparators/simulation")
    ap.add_argument("--ref", default="results/simulation")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for hide in [(), ("L1", "R1")]:
        tag = "hidden-" + "-".join(hide) if hide else "full-lr"
        ref_json = json.load(open(os.path.join(a.ref, f"all_results_{tag}.json")))
        allr = []
        for s in a.seeds:
            r, pg = run_seed(s, hide, ref_json)
            pg.to_csv(os.path.join(a.out, f"gene_scores_seed{s}_{tag}.csv"), index=False)
            print(json.dumps({k: v for k, v in r.items() if k != "metrics"}), flush=True)
            for m, v in r["metrics"].items():
                if "Selectivity" in m:
                    continue
                print(f"  {m:55s} AUPRC={v['auprc']:.3f} P@10={v['p@10']:.2f} ranks={v['ranks']}", flush=True)
            allr.append(r)
            json.dump(allr, open(os.path.join(a.out, f"all_results_{tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
