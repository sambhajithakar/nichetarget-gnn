"""NicheTarget-GNN, hybrid and baselines on 10x Xenium FFPE human breast cancer.

Data: Janesick et al. (Nat Commun 2023), 10x Genomics public datasets
"Xenium_FFPE_Human_Breast_Cancer_Rep1/Rep2" (313-gene panel).

Pre-specified design (fixed before looking at any validation resource)
---------------------------------------------------------------------
* Cell types: Xenium graph-based clusters mapped to lineages with canonical
  markers (CLUSTER_MAP below; printed marker table in the log).
* Malignant states: invasive-tumour clusters = target state; DCIS clusters =
  reference state. Read-out = the 15 genes most up-regulated in the target
  state (same rule as the simulation).
* Graph: k = 10 nearest neighbours; niches = k-means (5) on the cell-type
  composition of the 15 nearest neighbours.
* Model: same hyper-parameters as the simulation (d = 128, 2 layers, AdamW
  2e-3, 15% masking, read-out weight 5). Because a section has ~165k cells,
  training uses spatial tiles (mini-batches of >= BATCH_CELLS cells) for
  N_STEPS steps.
* No matched normal tissue exists for this section, so the selectivity term is
  not applied (efficacy-only ranking for every method).
* Hybrid = mean of the rank of the GNN ensemble score and the rank of the
  niche-stratified LR-communication score (as in the simulation).

    python scripts/run_xenium.py --data /path/rep1.pkl --lr lr_pairs_human.csv --out results/xenium_rep1
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nichetarget import np_graph as NG, np_model as NM, np_counterfactual as NC, np_baselines as NB  # noqa: E402

N_READOUT, K_GRAPH, K_NICHE, N_NICHES = 15, 10, 15, 5
TILE_UM, BATCH_CELLS = 400.0, 6000
N_SCAN = 10000          # target cells used to estimate efficacy (see run_members)

# Xenium graph-based clusters (Rep1) -> lineage, assigned from the marker table
LINEAGES = ["malignant", "myoepithelial", "fibroblast", "perivascular", "endothelial", "macrophage",
            "T cell", "B cell", "plasma cell", "mast cell", "immune-stromal mixed"]
TARGET_CLUSTERS_REP1 = [1, 7, 17, 18]        # invasive tumour (CEACAM6-low, FASN/ABCC11-high)
REF_CLUSTERS_REP1 = [3, 12, 16, 19]          # DCIS (CEACAM6/SERPINA3/TACSTD2-high)
CLUSTER_MAP_REP1 = {1: 0, 3: 0, 7: 0, 9: 0, 12: 0, 16: 0, 17: 0, 18: 0, 19: 0, 13: 1, 2: 2, 11: 3, 8: 4,
                    4: 5, 5: 6, 6: 6, 15: 7, 14: 8, 20: 9, 10: 10}


def marker_lineage(X, genes, cluster):
    """Cluster -> lineage by canonical markers (for sections whose cluster ids differ from Rep1).

    Each marker's cluster means are z-scored across clusters, so a marker that
    spills into many clusters (e.g. LUM) does not dominate; the lineage with
    the highest mean z-score wins.
    """
    gi = {g: i for i, g in enumerate(genes)}
    sets = {0: ["EPCAM", "KRT7", "KRT8", "ERBB2", "FOXA1"], 1: ["KRT5", "KRT14", "KRT15"],
            2: ["LUM", "POSTN", "PDGFRA", "DPT"], 3: ["MYH11", "PDGFRB", "ACTG2"], 4: ["PECAM1", "VWF", "CLEC14A"],
            5: ["CD68", "CD163", "C1QA"], 6: ["CD3E", "CD8A", "TRAC"], 7: ["MS4A1", "CD79A", "BANK1"],
            8: ["MZB1", "TNFRSF17", "DERL3"], 9: ["CPA3", "TPSAB1", "KIT"]}
    cl = np.unique(cluster)
    M = np.stack([X[cluster == c].mean(0) for c in cl])
    Z = (M - M.mean(0)) / (M.std(0) + 1e-9)
    out = {}
    for i, c in enumerate(cl):
        # markers absent from a smaller panel (e.g. KRT15, DERL3 on the 280-gene base panel) are skipped
        sc = {t: Z[i, [gi[g] for g in gs if g in gi]].mean() for t, gs in sets.items()}
        out[int(c)] = max(sc, key=sc.get)
    return out


def tumour_states_by_reference(X, genes, cluster, lineage_of):
    """Invasive vs DCIS for malignant clusters, from the DCIS markers found in Rep1."""
    gi = {g: i for i, g in enumerate(genes)}
    dcis = [gi[g] for g in ["CEACAM6", "SERPINA3", "TACSTD2"]]
    mal = [c for c, l in lineage_of.items() if l == 0]
    score = {c: X[cluster == c][:, dcis].mean() for c in mal}
    thr = np.median(list(score.values()))
    return [c for c in mal if score[c] < thr], [c for c in mal if score[c] >= thr]


def metrics_free_table(genes, scores: dict):
    df = pd.DataFrame({"gene": genes})
    for k, v in scores.items():
        df[k] = v
    return df


def loss_from_log(path):
    """Training-loss checkpoints (every 25 steps) printed by a previous run."""
    out = []
    if os.path.exists(path):
        for line in open(path):
            if line.startswith("epoch"):
                out.append(float(line.split()[-1]))
    return out


def run_members(a, members, g, genes, coords, target_cells, ref_cells, niche, info):
    """Train and scan the requested ensemble members; load the others from disk.

    Returns None if any member is still missing (or with --no-combine).
    """
    tiles = NG.spatial_tiles(coords, TILE_UM)
    # efficacy is estimated on the target cells of randomly ordered spatial tiles (fixed seed) until
    # >= N_SCAN cells; whole tiles keep the 2-hop message-passing halo small
    is_t = np.zeros(g.n, bool); is_t[target_cells] = True
    picked, tot = [], 0
    for t in np.random.default_rng(0).permutation(len(tiles)):
        c = tiles[t][is_t[tiles[t]]]
        if len(c):
            picked.append(c); tot += len(c)
        if tot >= N_SCAN:
            break
    scan_cells = np.sort(np.concatenate(picked))
    info["n_scan_cells"] = int(len(scan_cells))
    eff_models, glob_models, per_niche_models, hist_all = [], [], [], []
    missing = False
    for e in range(a.n_ensemble):
        s_e = a.seed * 100 + e
        rows_path = os.path.join(a.out, f"model{e}_rows.pkl")
        if e not in members and os.path.exists(rows_path):
            rows, hist = pickle.load(open(rows_path, "rb"))
        elif e not in members:
            missing = True
            continue
        else:
            rng = np.random.default_rng(s_e)

            def batches(step, rng=rng):
                order = rng.permutation(len(tiles))
                nodes, tot = [], 0
                for t in order:
                    nodes.append(tiles[t]); tot += len(tiles[t])
                    if tot >= BATCH_CELLS:
                        break
                return [NG.subgraph(g, np.sort(np.concatenate(nodes)))]

            model = NM.NicheTargetGNN(len(genes), len(LINEAGES), len(g.lr_idx), seed=s_e)
            params_path = os.path.join(a.out, f"model{e}_params.pkl")
            if os.path.exists(params_path):          # resume: trained earlier, scan was interrupted
                model.load(pickle.load(open(params_path, "rb")))
                hist = loss_from_log(os.path.join(a.out, f"log_member{e}.txt"))
                print(f"model {e} loaded from {params_path}", flush=True)
            else:
                tm = time.time()
                hist = NM.train_model(model, g, epochs=a.steps, seed=s_e, batches=batches, verbose=True)
                print(f"model {e} trained in {time.time() - tm:.0f}s", flush=True)
                with open(params_path, "wb") as f:
                    pickle.dump(model.state(), f)
            tm = time.time()
            cand = list(range(a.debug_genes)) if a.debug_genes else None
            rows, _ = NC.scan_targets(model, g, genes, scan_cells, ref_cells, niche, chunk=None, candidates=cand,
                                      axis_cells=target_cells)
            if cand is not None:
                rows = rows + [(genes[j], 0.0, {}) for j in range(a.debug_genes, len(genes))]
            print(f"model {e} scanned in {time.time() - tm:.0f}s", flush=True)
            with open(rows_path, "wb") as f:
                pickle.dump((rows, hist), f)
        eff_models.append(np.array([NC.niche_max(r[2]) for r in rows]))
        glob_models.append(np.array([r[1] for r in rows]))
        per_niche_models.append({j: r[2] for j, r in enumerate(rows)})
        hist_all.append(hist)
    if missing or a.no_combine:
        print("members done; run again without --no-combine to combine", flush=True)
        return None
    return eff_models, glob_models, per_niche_models, hist_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--lr", required=True, help="CSV with ligand,receptor columns")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-ensemble", type=int, default=3)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--section", default="rep1", choices=["rep1", "other"])
    ap.add_argument("--hide-lr", nargs="*", default=[], help="genes whose LR pairs are removed (robustness)")
    ap.add_argument("--debug-genes", type=int, default=0, help="scan only the first N genes (smoke test)")
    ap.add_argument("--members", type=int, nargs="*", default=None,
                    help="ensemble members trained in this process (default: all); others are loaded if saved")
    ap.add_argument("--no-combine", action="store_true", help="only train/scan the given members")
    a = ap.parse_args()
    members = list(range(a.n_ensemble)) if a.members is None else a.members
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    d = pickle.load(open(a.data, "rb"))
    X, genes, coords, cluster = d["X"], list(d["genes"]), d["coords"], d["cluster"]
    gi = {g: i for i, g in enumerate(genes)}

    if a.section == "rep1":
        lineage_of, tgt_cl, ref_cl = CLUSTER_MAP_REP1, TARGET_CLUSTERS_REP1, REF_CLUSTERS_REP1
    else:
        lineage_of = marker_lineage(X, genes, cluster)
        tgt_cl, ref_cl = tumour_states_by_reference(X, genes, cluster, lineage_of)
    cell_type = np.array([lineage_of.get(int(c), 10) for c in cluster])
    target_cells = np.where(np.isin(cluster, tgt_cl))[0]
    ref_cells = np.where(np.isin(cluster, ref_cl))[0]
    malignant = cell_type == 0
    mal_cells = np.where(np.isin(cluster, tgt_cl + ref_cl))[0]
    malignant_states = np.zeros(len(X), bool)
    malignant_states[mal_cells] = True

    de = NB.de_score(X, target_cells, ref_cells)
    readout_idx = np.argsort(de)[::-1][:N_READOUT]
    readout = [genes[i] for i in readout_idx]
    programme = X[:, readout_idx].mean(1)

    lr = pd.read_csv(a.lr)
    lr_all = [(l, r) for l, r in zip(lr.ligand, lr.receptor) if l in gi and r in gi and l != r]
    lr_all = sorted(set(lr_all))
    lr_known = [p for p in lr_all if p[0] not in a.hide_lr and p[1] not in a.hide_lr]

    info = {"n_cells": int(len(X)), "n_genes": len(genes), "n_target": int(len(target_cells)),
            "n_reference": int(len(ref_cells)), "target_clusters": tgt_cl, "reference_clusters": ref_cl,
            "lineage_of_cluster": {int(k): LINEAGES[v] for k, v in lineage_of.items()},
            "lineage_counts": {LINEAGES[t]: int((cell_type == t).sum()) for t in range(len(LINEAGES))},
            "readout": readout, "readout_de": [float(de[i]) for i in readout_idx],
            "n_lr_pairs_in_panel": len(lr_all), "lr_pairs_used": len(lr_known), "hidden_lr": a.hide_lr}
    print(json.dumps(info), flush=True)

    dvals, nbr = NG.knn(coords, K_GRAPH)
    scale = float(np.quantile(dvals, 0.999))
    g = NG.build_graph(X, coords, cell_type, len(LINEAGES), genes, lr_known, readout, k=K_GRAPH,
                       k_niche=K_NICHE, dist_scale=scale)
    g.edge_dist = np.clip(g.edge_dist, 0, 1).astype(np.float32)
    niche = NG.kmeans(g.niche_comp, N_NICHES, seed=a.seed, n_init=5)
    comp = pd.DataFrame(g.niche_comp).groupby(niche).mean()
    comp.columns = LINEAGES
    info["niche_composition"] = comp.round(3).to_dict(orient="index")
    info["niche_target_cells"] = {int(k): int(v) for k, v in zip(*np.unique(niche[target_cells], return_counts=True))}
    print(comp.round(2).to_string(), flush=True)

    if a.no_combine:
        run_members(a, members, g, genes, coords, target_cells, ref_cells, niche, info)
        return

    # ---- baselines ------------------------------------------------------------
    scores = {}
    tb = time.time()
    scores["DE"] = de
    scores["DE (niche-stratified)"] = NB.de_niche_max(X, target_cells, ref_cells, niche)
    scores["NeighbourhoodCorr"] = NB.neighbourhood_correlation(X, nbr, malignant_states, programme)
    scores["NeighbourhoodCorr (niche-stratified)"] = NB.niche_stratified(
        NB.neighbourhood_correlation, mal_cells, niche, X=X, nbr=nbr, malignant=malignant_states, programme=programme)
    scores["LR-communication"] = NB.lr_communication(X, nbr, malignant_states, programme, genes, lr_known)
    scores["LR-communication (niche-stratified)"] = NB.niche_stratified(
        NB.lr_communication, mal_cells, niche, X=X, nbr=nbr, malignant=malignant_states, programme=programme,
        genes=genes, lr_pairs=lr_known)
    info["baseline_runtime_s"] = time.time() - tb
    print("baselines done", time.time() - t0, flush=True)

    # ---- NicheTarget-GNN ensemble -----------------------------------------------
    res = run_members(a, members, g, genes, coords, target_cells, ref_cells, niche, info)
    if res is None:
        return
    eff_models, glob_models, per_niche_models, hist_all = res
    n_layers = 2

    # readout R2 of the first model on target+reference cells (2-hop exact predictions)
    cells = np.concatenate([target_cells, ref_cells])
    model0 = NM.NicheTargetGNN(len(genes), len(LINEAGES), len(g.lr_idx), seed=0)
    model0.load(pickle.load(open(os.path.join(a.out, "model0_params.pkl"), "rb")))
    pred = np.zeros((len(cells), len(readout_idx)), np.float32)
    for i in range(0, len(cells), 8000):          # chunked to bound memory
        cc = cells[i:i + 8000]
        nd = NG.hops(g, cc, n_layers)
        sgc = NG.subgraph(g, nd)
        pred[i:i + 8000] = model0.predict(NM.make_input(sgc.x, sgc.readout_idx), sgc)[np.searchsorted(nd, cc)][:, readout_idx]
    yt = X[cells][:, readout_idx]
    info["readout_R2_model0"] = float(1 - ((yt - pred) ** 2).sum() / ((yt - yt.mean(0)) ** 2).sum())

    per_niche_mean = {}
    for j in range(len(genes)):
        keys = set().union(*[pm[j].keys() for pm in per_niche_models])
        per_niche_mean[j] = {n: float(np.mean([pm[j].get(n, 0.0) for pm in per_niche_models])) for n in keys}
    eff_ens = np.array([max(per_niche_mean[j].values()) if per_niche_mean[j] else 0.0 for j in range(len(genes))])
    scores["NicheTarget-GNN (single model)"] = NC.combine_scores(eff_models[0], None)
    scores["NicheTarget-GNN (global efficacy)"] = NC.combine_scores(glob_models[0], None)
    scores[f"NicheTarget-GNN (ensemble of {a.n_ensemble})"] = NC.combine_scores(eff_ens, None)
    G_ = len(genes)
    gnn_ens = scores[f"NicheTarget-GNN (ensemble of {a.n_ensemble})"]
    scores["Hybrid (GNN ensemble + LR-communication)"] = (0.5 * NG.rankdata(gnn_ens) / G_
                                                         + 0.5 * NG.rankdata(scores["LR-communication (niche-stratified)"]) / G_)
    table = metrics_free_table(genes, scores)
    table["is_readout"] = table.gene.isin(readout)
    table["in_lr_database"] = table.gene.isin({x for p in lr_known for x in p})
    table["mean_expr_target"] = X[target_cells].mean(0)
    table["mean_expr_malignant"] = X[malignant].mean(0)
    for t in range(len(LINEAGES)):
        table[f"mean_expr_{LINEAGES[t].replace(' ', '_')}"] = X[cell_type == t].mean(0)
    for nn in sorted(set(niche[target_cells].tolist())):
        table[f"efficacy_niche{nn}"] = [per_niche_mean[j].get(nn, np.nan) for j in range(G_)]
    table["best_niche"] = [max(per_niche_mean[j], key=per_niche_mean[j].get) if per_niche_mean[j] else -1 for j in range(G_)]
    table.to_csv(os.path.join(a.out, "gene_scores.csv"), index=False)
    np.save(os.path.join(a.out, "niche.npy"), niche)
    np.save(os.path.join(a.out, "cell_type.npy"), cell_type)
    info["train_loss_final"] = [h[-1] for h in hist_all]
    info["train_loss_curves"] = hist_all
    info["runtime_s"] = time.time() - t0
    json.dump(info, open(os.path.join(a.out, "run_info.json"), "w"), indent=1)
    top = table[~table.is_readout].sort_values("Hybrid (GNN ensemble + LR-communication)", ascending=False).head(25)
    print(top[["gene", "Hybrid (GNN ensemble + LR-communication)", f"NicheTarget-GNN (ensemble of {a.n_ensemble})",
               "LR-communication (niche-stratified)"]].to_string(), flush=True)


if __name__ == "__main__":
    main()
