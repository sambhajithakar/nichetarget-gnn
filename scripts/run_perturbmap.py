"""Perturb-map validation: does in-silico knockout recover real CRISPR-knockout lesion phenotypes?

Data: Dhainaut et al., Cell 2022 (GEO GSE193460). Four Visium sections of KP
mouse lung adenocarcinoma in which lesions carry Pro-Code-barcoded CRISPR
knockouts. The GEO spot annotation labels Tgfbr2-KO, Ifngr2-KO and Jak2-KO
lesions plus other lesions (``KP_<section>-<k>``, used here as comparison
lesions).

Pre-specified protocol
----------------------
1. log1p(CP10k). Genes: top 1000 highly variable genes among comparison-lesion
   spots, plus the knockout genes and ligand/receptor genes of the mouse LR
   database that are expressed in >= 5% of spots.
2. Read-out genes (always hidden from the input, predicted only from other
   genes and neighbours): the top 300 of those HVGs plus fixed IFN-gamma and
   TGF-beta response gene lists; knockout genes are never read-out genes and
   LR pairs containing a read-out gene are dropped.
3. Graph: 6 nearest spots within each section. Spots of every annotated
   knockout lesion are removed before training, so the model never sees a
   knockout phenotype. Spot classes for the niche descriptor: lesion,
   lesion periphery, other tumour, normal lung.
4. Models: NicheTarget-GNN (full) and the same model without spatial message
   passing, 3 seeds each, same hyper-parameters as in the simulation.
5. In-silico knockout of gene j = set j to 0 in comparison-lesion spots of the
   sections carrying that knockout (cell-intrinsic, not systemic); predicted
   effect = mean change of the read-out genes over those spots.
6. Observed effect of knockout k = mean over sections carrying k of
   (mean in k-KO lesion spots - mean in comparison-lesion spots), read-out genes.
7. Metrics per knockout with >= 30 spots: Spearman(predicted, observed); the
   rank of the true knockout gene among all candidate genes (input genes
   expressed in >= 10% of comparison spots) when candidates are ordered by how
   well their predicted effect matches the observed one; and, for Ifngr2, the
   fraction of IFN-gamma response genes predicted to go down. A co-expression
   (linear regression) predictor is the non-learned baseline.

    python scripts/run_perturbmap.py --npz /path/npz --raw /path/raw --lr lr_pairs_mouse.csv --out results/perturbmap
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nichetarget import np_graph as NG, np_model as NM  # noqa: E402

KO_GENES = {"Tgfbr2": "Tgfbr2", "Ifngr2": "Ifngr2", "Jak2": "Jak2"}
IFNG_RESPONSE = ["Cxcl9", "Cxcl10", "Stat1", "Irf1", "Gbp2", "Gbp3", "Gbp5", "Gbp7", "Ido1", "Cd274", "Psmb8",
                 "Psmb9", "Tap1", "H2-Aa", "H2-Ab1", "H2-Eb1", "Ciita", "Iigp1", "Igtp", "Irgm1", "Socs1"]
TGFB_RESPONSE = ["Serpine1", "Smad7", "Skil", "Tgfbi", "Ccn2", "Fn1", "Col1a1", "Col1a2", "Col3a1", "Postn",
                 "Ltbp2", "Pmepa1", "Junb", "Lrrc15", "Timp1"]
N_HVG, N_READOUT, K_SPOT, MIN_KO_SPOTS = 1000, 300, 6, 30
SPOT_CLASSES = ["lesion", "periphery", "other tumour", "normal"]


def load_section(npz_path, raw_dir):
    z = np.load(npz_path)
    G, B = z["shape"]
    X = np.zeros((B, G), dtype=np.float32)
    indptr = z["indptr"]
    cols = np.repeat(np.arange(B), np.diff(indptr))
    X[cols, z["indices"]] = z["data"]
    stem = os.path.basename(npz_path).replace(".npz", "")
    ann = pd.read_csv(os.path.join(raw_dir, f"{stem}_spot_annotation.csv.gz"))
    pos = pd.read_csv(os.path.join(raw_dir, f"{stem}_tissue_positions_list.csv.gz"), header=None,
                      names=["barcode", "in_tissue", "row", "col", "pxl_row", "pxl_col"]).set_index("barcode")
    bc = pd.Index(z["barcodes"].astype(str))
    keep = bc.get_indexer(ann.barcode)
    X = X[keep]
    coords = pos.loc[ann.barcode, ["pxl_col", "pxl_row"]].to_numpy(float)
    return X, list(z["feature_name"].astype(str)), coords, ann.reset_index(drop=True), stem


def spot_class(ann):
    ph = ann.phenotypes.fillna("")
    c = np.full(len(ann), 3)
    c[(ann.kmeans == "tumor").to_numpy()] = 2
    c[(ph == "periphery").to_numpy()] = 1
    c[(ph.str.len() > 0) & (ph != "periphery")] = 0
    return c


def ko_label(ph):
    for k in KO_GENES:
        if isinstance(ph, str) and ph.startswith(k + "_"):
            return k
    return None


def spearman(a, b):
    return NG.spearman(a, b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--lr", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--variants", nargs="+", default=["full", "no_spatial"])
    ap.add_argument("--combine-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    secs = [load_section(p, a.raw) for p in sorted(glob.glob(os.path.join(a.npz, "*.npz")))]
    genes_all = secs[0][1]
    assert all(s[1] == genes_all for s in secs)
    counts = np.concatenate([s[0] for s in secs])
    ann = pd.concat([s[3].assign(section=s[4]) for s in secs], ignore_index=True)
    coords = np.concatenate([s[2] + np.array([1e6 * i, 0]) for i, s in enumerate(secs)])   # sections never connect
    del secs
    tot = counts.sum(1, keepdims=True)
    # keep genes detected in >= 5% of spots (plus the knockout genes) before densifying further
    detected = (counts > 0).mean(0) >= 0.05
    detected[[genes_all.index(g) for g in KO_GENES.values()]] = True
    keep_g = np.where(detected)[0]
    counts = counts[:, keep_g]
    genes_all = [genes_all[i] for i in keep_g]
    Xall = np.log1p(counts / np.maximum(tot, 1) * 1e4).astype(np.float32)
    cls = spot_class(ann)
    ko = ann.phenotypes.map(ko_label).to_numpy()
    is_ko = pd.notna(ko)
    is_cmp = (cls == 0) & ~is_ko

    # ---- gene selection (comparison-lesion spots only; never uses knockout spots)
    expressed = (counts > 0).mean(0) >= 0.05
    v = Xall[is_cmp].var(0)
    v[~expressed] = -1
    hvg = list(np.argsort(v)[::-1][:N_HVG])
    gi_all = {g: i for i, g in enumerate(genes_all)}
    lr = pd.read_csv(a.lr)
    lr_genes = set(lr.ligand) | set(lr.receptor)
    extra = [gi_all[g] for g in genes_all if g in lr_genes and expressed[gi_all[g]]]
    ko_idx = [gi_all[g] for g in KO_GENES.values()]
    path_idx = [gi_all[g] for g in IFNG_RESPONSE + TGFB_RESPONSE if g in gi_all and expressed[gi_all[g]]]
    sel = sorted(set(hvg) | set(extra) | set(ko_idx) | set(path_idx))
    genes = [genes_all[i] for i in sel]
    gi = {g: i for i, g in enumerate(genes)}
    X = Xall[:, sel]
    readout = [genes_all[i] for i in hvg[:N_READOUT]] + [genes_all[i] for i in path_idx]
    readout = [g for g in dict.fromkeys(readout) if g not in KO_GENES.values()]
    rset = set(readout)
    lr_pairs = sorted({(l, r) for l, r in zip(lr.ligand, lr.receptor)
                       if l in gi and r in gi and l != r and l not in rset and r not in rset})

    # ---- training graph without knockout lesions
    train = np.where(~is_ko)[0]
    g = NG.build_graph(X[train], coords[train], cls[train], len(SPOT_CLASSES), genes, lr_pairs, readout,
                       k=K_SPOT, k_niche=K_SPOT)
    pos_in_train = -np.ones(len(X), int)
    pos_in_train[train] = np.arange(len(train))
    info = {"n_spots": int(len(X)), "n_train_spots": int(len(train)), "n_genes_model": len(genes),
            "n_readout": len(readout), "n_lr_pairs": len(lr_pairs),
            "ko_spots": {k: int((ko == k).sum()) for k in KO_GENES},
            "comparison_spots": int(is_cmp.sum()),
            "ifng_genes_in_readout": [x for x in IFNG_RESPONSE if x in rset],
            "tgfb_genes_in_readout": [x for x in TGFB_RESPONSE if x in rset]}
    print(json.dumps(info), flush=True)

    r_idx = np.array([gi[x] for x in readout])
    cmp_sec = ann.section.to_numpy()
    kos = [k for k in KO_GENES if (ko == k).sum() >= MIN_KO_SPOTS]
    observed, cmp_cells = {}, {}
    for k in kos:
        secs_k = sorted(set(cmp_sec[ko == k]))
        diffs = [X[(ko == k) & (cmp_sec == s)][:, r_idx].mean(0) - X[is_cmp & (cmp_sec == s)][:, r_idx].mean(0) for s in secs_k]
        observed[k] = np.mean(diffs, 0)
        cmp_cells[k] = pos_in_train[np.where(is_cmp & np.isin(cmp_sec, secs_k))[0]]
    frac_cmp = {k: (X[np.where(is_cmp & np.isin(cmp_sec, sorted(set(cmp_sec[ko == k]))))[0]] > 0).mean(0) for k in kos}
    candidates = {k: [j for j in range(len(genes)) if genes[j] not in rset and frac_cmp[k][j] >= 0.10] for k in kos}
    info["n_candidates"] = {k: len(v) for k, v in candidates.items()}
    info["observed_ifng_mean_change"] = {k: float(observed[k][[readout.index(x) for x in info["ifng_genes_in_readout"]]].mean()) for k in kos}

    def predicted_effects(model, k):
        # exact predictions for the comparison spots need only their message-passing neighbourhood
        nodes = NG.hops(g, cmp_cells[k], model.n_layers if model.use_spatial else 0)
        sg = NG.subgraph(g, nodes)
        cells = np.searchsorted(nodes, cmp_cells[k])
        x_in = NM.make_input(sg.x, sg.readout_idx)
        base = model.predict(x_in, sg)[cells][:, sg.readout_idx]
        out = {}
        for j in candidates[k]:
            x_cf = x_in.copy()
            x_cf[cells, j] = 0.0
            out[j] = (model.predict(x_cf, sg)[cells][:, sg.readout_idx] - base).mean(0)
        return out

    # ---- train / scan
    for var in a.variants:
        for s in a.seeds:
            path = os.path.join(a.out, f"effects_{var}_seed{s}.pkl")
            if os.path.exists(path) or a.combine_only:
                continue
            model = NM.NicheTargetGNN(len(genes), len(SPOT_CLASSES), len(g.lr_idx), seed=s,
                                      use_spatial=(var == "full"))
            tm = time.time()
            hist = NM.train_model(model, g, epochs=a.epochs, seed=s, verbose=True)
            eff = {k: predicted_effects(model, k) for k in kos}
            pickle.dump({"effects": eff, "hist": hist, "train_s": time.time() - tm}, open(path, "wb"))
            print(f"{var} seed {s} done in {time.time() - tm:.0f}s", flush=True)

    # ---- co-expression baseline: linear regression of each read-out gene on candidate j
    def coexpr_effects(k):
        cells = cmp_cells[k]
        Xc = g.x[cells]
        out = {}
        R = Xc[:, g.readout_idx]
        for j in candidates[k]:
            xj = Xc[:, j]
            vj = xj.var()
            beta = ((xj - xj.mean())[:, None] * (R - R.mean(0))).mean(0) / (vj + 1e-12)
            out[j] = -beta * xj.mean()
        return out

    results = {}
    files = sorted(glob.glob(os.path.join(a.out, "effects_*_seed*.pkl")))
    if not files:
        return
    methods = {}
    for var in ["full", "no_spatial"]:
        fs = [f for f in files if f"effects_{var}_seed" in f]
        if fs:
            effs = [pickle.load(open(f, "rb"))["effects"] for f in fs]
            methods[{"full": "NicheTarget-GNN", "no_spatial": "w/o spatial message passing"}[var]] = \
                {k: {j: np.mean([e[k][j] for e in effs], 0) for j in candidates[k]} for k in kos}
            methods[{"full": "NicheTarget-GNN", "no_spatial": "w/o spatial message passing"}[var] + " (n models)"] = len(fs)
    methods["Co-expression regression"] = {k: coexpr_effects(k) for k in kos}
    rows = []
    ifng_r = [readout.index(x) for x in info["ifng_genes_in_readout"]]
    for m, eff in methods.items():
        if not isinstance(eff, dict):
            continue
        for k in kos:
            true_j = gi[KO_GENES[k]]
            sims = {j: spearman(eff[k][j], observed[k]) for j in candidates[k]}
            sims = {j: (0.0 if np.isnan(v) else v) for j, v in sims.items()}
            order = sorted(sims, key=lambda j: -sims[j])
            rank = order.index(true_j) + 1 if true_j in sims else None
            row = {"method": m, "knockout": k, "n_ko_spots": int((ko == k).sum()),
                   "spearman_pred_vs_obs": sims.get(true_j), "rank_of_true_gene": rank,
                   "n_candidates": len(candidates[k]),
                   "percentile": (1 - (rank - 1) / len(candidates[k])) if rank else None,
                   "top5_genes": [genes[j] for j in order[:5]]}
            if ifng_r and true_j in eff[k]:
                row["frac_ifng_genes_predicted_down"] = float((eff[k][true_j][ifng_r] < 0).mean())
                row["mean_pred_ifng_change"] = float(eff[k][true_j][ifng_r].mean())
            rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(a.out, "perturbmap_validation.csv"), index=False)
    info["n_models"] = {m: v for m, v in methods.items() if not isinstance(v, dict)}
    info["runtime_s"] = time.time() - t0
    json.dump(info, open(os.path.join(a.out, "run_info.json"), "w"), indent=1)
    pd.DataFrame({"gene": readout, **{f"observed_{k}": observed[k] for k in kos},
                  **{f"pred_{m}_{k}": methods[m][k][gi[KO_GENES[k]]] for m in methods if isinstance(methods[m], dict)
                     for k in kos if gi[KO_GENES[k]] in methods[m][k]}}).to_csv(
        os.path.join(a.out, "readout_effects.csv"), index=False)
    print(res.to_string(), flush=True)


if __name__ == "__main__":
    main()
