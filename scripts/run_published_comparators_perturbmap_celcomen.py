"""Celcomen / SimComen (published in-silico perturbation model) on Perturb-map, with the paper's metrics.

Tool: Celcomen (Teichlab/celcomen; Megas et al.), installed from the GitHub source
(commit recorded in results/published_comparators/perturbmap_celcomen/run_info.json). Celcomen learns
intercellular (graph convolution, G x G) and intracellular (G x G) gene-gene interaction matrices from
unperturbed spatial data; SimComen takes these matrices, starts from a (perturbed) expression state and
relaxes it by gradient steps on the same energy, which yields the predicted counterfactual.

Data and evaluation follow the pre-specified protocol of scripts/run_perturbmap.py exactly (the data
preparation below is a verbatim copy of that script; it is checked against results/perturbmap/run_info.json):
same 1,759 genes, same 318 read-out genes, all knockout-lesion spots removed before training, same
candidates (1,371 for Tgfbr2, 1,347 for Ifngr2), same observed effects, same Spearman similarity and
rank-of-true-knockout metric, same bootstrap CI as scripts/stats_supplement.py.

Celcomen-specific choices (from the Celcomen Xenium tutorial; not tuned):
  * model genes = the same 1,759 genes; Celcomen models all genes jointly (it has no hidden read-out),
    so read-out genes are part of its state; this differs from NicheTarget-GNN, which never sees them as input.
  * training: published datareader ``get_dataset_loaders`` (one graph per Visium section, k = 6 nearest
    spots, which is the k of the paper's graph), published ``train`` with lr = 0.1, zmft = 0.1, 200 epochs,
    random symmetric initialisation normalised with ``normalize_g2g`` (tutorial settings).
  * counterfactual: for knockout k, the simulated tissue is the comparison-lesion spots of the sections
    carrying k plus their 1-hop neighbours in the training graph (edges = k = 6 nearest spots, as in training).
    Expression is L2-normalised per spot (tutorial). Knockout of candidate j = expression of j set to 0 in
    the comparison-lesion spots before normalisation (tutorial). SimComen: lr = 1e-3, momentum 0, zmft = 0.1,
    50 epochs (tutorial). Predicted effect of j = mean over the comparison-lesion spots of
    [SimComen(perturbed) - SimComen(unperturbed)] for the read-out genes (both runs use identical settings).
  * speed: SimComen's ``calc_gex`` is a Python double loop over genes (O(G^2) tensor operations, about
    1.5 million per step at G = 1,759) and the loss uses trace(msg @ gex^T), which forms an N x N matrix.
    The published SimComen also cannot run at this size: its autograd graph holds ~G^2/2 in-place slice
    operations per step and was killed by the out-of-memory killer (> 12 GB) even for a 150-spot tissue.
    For the counterfactual scan only, we therefore use (i) a subclass whose ``calc_gex`` computes the same
    product of sines by a cumulative product, (ii) the trace written as an element-wise sum, and (iii) a
    re-ordered ``calc_sphex`` that performs the identical float32 operations per element. (iii) is checked to
    be bit-identical and (i)+(ii) to agree to ~1e-7 with the published code on real data (``--check-real``).

    python scripts/run_published_comparators_perturbmap_celcomen.py --npz .../npz --raw .../raw \
        --lr lr_pairs_mouse.csv --out results/published_comparators/perturbmap_celcomen --seeds 1
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from nichetarget import np_graph as NG  # noqa: E402

# ---- verbatim constants / helpers from scripts/run_perturbmap.py ---------------------------------
KO_GENES = {"Tgfbr2": "Tgfbr2", "Ifngr2": "Ifngr2", "Jak2": "Jak2"}
IFNG_RESPONSE = ["Cxcl9", "Cxcl10", "Stat1", "Irf1", "Gbp2", "Gbp3", "Gbp5", "Gbp7", "Ido1", "Cd274", "Psmb8",
                 "Psmb9", "Tap1", "H2-Aa", "H2-Ab1", "H2-Eb1", "Ciita", "Iigp1", "Igtp", "Irgm1", "Socs1"]
TGFB_RESPONSE = ["Serpine1", "Smad7", "Skil", "Tgfbi", "Ccn2", "Fn1", "Col1a1", "Col1a2", "Col3a1", "Postn",
                 "Ltbp2", "Pmepa1", "Junb", "Lrrc15", "Timp1"]
N_HVG, N_READOUT, K_SPOT, MIN_KO_SPOTS = 1000, 300, 6, 30


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


def prepare_data(npz, raw, lr_path):
    """Verbatim data preparation of scripts/run_perturbmap.py::main (up to the observed effects)."""
    secs = [load_section(p, raw) for p in sorted(glob.glob(os.path.join(npz, "*.npz")))]
    genes_all = secs[0][1]
    counts = np.concatenate([s[0] for s in secs])
    ann = pd.concat([s[3].assign(section=s[4]) for s in secs], ignore_index=True)
    coords = np.concatenate([s[2] + np.array([1e6 * i, 0]) for i, s in enumerate(secs)])
    del secs
    tot = counts.sum(1, keepdims=True)
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
    expressed = (counts > 0).mean(0) >= 0.05
    v = Xall[is_cmp].var(0)
    v[~expressed] = -1
    hvg = list(np.argsort(v)[::-1][:N_HVG])
    gi_all = {g: i for i, g in enumerate(genes_all)}
    lr = pd.read_csv(lr_path)
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
    train = np.where(~is_ko)[0]
    r_idx = np.array([gi[x] for x in readout])
    cmp_sec = ann.section.to_numpy()
    kos = [k for k in KO_GENES if (ko == k).sum() >= MIN_KO_SPOTS]
    observed, cmp_spots = {}, {}
    for k in kos:
        secs_k = sorted(set(cmp_sec[ko == k]))
        diffs = [X[(ko == k) & (cmp_sec == s)][:, r_idx].mean(0) - X[is_cmp & (cmp_sec == s)][:, r_idx].mean(0)
                 for s in secs_k]
        observed[k] = np.mean(diffs, 0)
        cmp_spots[k] = np.where(is_cmp & np.isin(cmp_sec, secs_k))[0]      # indices into all spots
    frac_cmp = {k: (X[np.where(is_cmp & np.isin(cmp_sec, sorted(set(cmp_sec[ko == k]))))[0]] > 0).mean(0)
                for k in kos}
    candidates = {k: [j for j in range(len(genes)) if genes[j] not in rset and frac_cmp[k][j] >= 0.10] for k in kos}
    ifng = [x for x in IFNG_RESPONSE if x in rset]
    return dict(X=X, genes=genes, gi=gi, readout=readout, r_idx=r_idx, coords=coords, ann=ann, train=train,
                kos=kos, observed=observed, cmp_spots=cmp_spots, candidates=candidates, ifng=ifng,
                cmp_sec=cmp_sec, ko=ko, n_spots=len(X))
# ---------------------------------------------------------------------------------------------------


def make_fast_simcomen():
    from celcomen.models.simcomen import simcomen

    class SimcomenFast(simcomen):
        """Published simcomen with an algebraically identical, vectorised calc_gex (cumulative product of sines)."""

        def calc_gex(self, sphex):
            # published loop: gex[:, j] = cos(t_j) * prod_{i<j} sin(t_i) for j < G-1, and
            # gex[:, G-1] = sin(t_{G-2}) * prod_{i<G-1} sin(t_i) (the last sine enters twice, as in the published code)
            s = torch.sin(sphex)
            c = torch.cos(sphex)
            ones = torch.ones((sphex.shape[0], 1), dtype=sphex.dtype, device=sphex.device)
            prefix = torch.cumprod(torch.cat([ones, s], 1), 1)     # prefix[:, j] = prod_{i<j} sin(t_i), j = 0..G-1
            gex = torch.cat([c * prefix[:, :-1], s[:, -1:] * prefix[:, -1:]], 1)
            return torch.nan_to_num(gex)

    return simcomen, SimcomenFast


def calc_sphex_fast(gex):
    """Re-ordering of the published calc_sphex loop that performs exactly the same float32 operations per element
    (x_i divided successively by sin(t_0), sin(t_1), ..., then arccos), but iterates over the divisor so that each
    step is one vectorised operation (O(G) Python steps instead of O(G^2)). Checked to be bit-identical to the
    published function on the real data (see equivalence_check_real_*.json)."""
    n_s = gex.shape[1] - 1
    work = gex[:, :n_s].clone().to(torch.float32)
    sphex = torch.zeros_like(work)
    for i in range(n_s):
        sphex[:, i] = torch.arccos(work[:, i])
        if i + 1 < n_s:
            work[:, i + 1:] /= torch.sin(sphex[:, i:i + 1])
    return sphex


def sim_loss(model, edge_index, zmft, fast=True):
    msg, msg_intra, log_z_mft = model(edge_index, 1)
    if fast:
        t1 = (msg * model.gex).sum()
        t2 = (msg_intra * model.gex).sum()
    else:
        t1 = torch.trace(torch.mm(msg, torch.t(model.gex)))
        t2 = torch.trace(torch.mm(msg_intra, torch.t(model.gex)))
    return -(-log_z_mft + zmft * t1 + zmft * t2)


def run_simcomen(cls, g2g, g2g_intra, expr_unit, edge_index, epochs, lr, zmft, seed, fast=True):
    from celcomen.utils.helpers import calc_sphex as calc_sphex_published
    calc_sphex = calc_sphex_fast if fast else calc_sphex_published
    n = g2g.shape[0]
    m = cls(input_dim=n, output_dim=n, n_neighbors=K_SPOT, seed=seed)
    m.set_g2g(g2g.clone())
    m.set_g2g_intra(g2g_intra.clone())
    m.set_sphex(calc_sphex(expr_unit).clone().detach())
    opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=0)
    m.train()
    orig = None
    for ep in range(epochs):
        loss = sim_loss(m, edge_index, zmft, fast)
        if ep == 0:
            orig = m.gex.clone().detach()
        loss.backward()
        opt.step()
        opt.zero_grad()
    with torch.no_grad():
        final = m.calc_gex(m.sphex)
    return orig, final


def unit(x):
    x = torch.as_tensor(x, dtype=torch.float32)
    nf = torch.sqrt(torch.pow(x, 2).sum(1)).reshape(-1, 1)
    return torch.div(x, torch.clamp(nf, min=1e-12))


def check_equivalence(out):
    """Published simcomen vs. the vectorised subclass on random data (small G) and on a real-size gene count."""
    simcomen, Fast = make_fast_simcomen()
    from celcomen.utils.helpers import normalize_g2g
    res = {}
    for G, N, ep in [(40, 60, 5), (300, 40, 2)]:
        rng = np.random.default_rng(0)
        x = unit(np.abs(rng.normal(size=(N, G))) * (rng.random((N, G)) > 0.3))
        g = torch.from_numpy(normalize_g2g(rng.uniform(size=(G, G)).astype("float32") * 0.05))
        gi = torch.from_numpy(normalize_g2g(rng.uniform(size=(G, G)).astype("float32") * 0.05))
        from sklearn.neighbors import kneighbors_graph
        pos = rng.random((N, 2))
        ei = torch.from_numpy(np.array(np.where(kneighbors_graph(pos, K_SPOT, include_self=False).toarray() == 1)))
        o1, f1 = run_simcomen(simcomen, g, gi, x, ei, ep, 1e-3, 0.1, 0, fast=False)
        o2, f2 = run_simcomen(Fast, g, gi, x, ei, ep, 1e-3, 0.1, 0, fast=True)
        res[f"G{G}_N{N}_epochs{ep}"] = {"max_abs_diff_initial_gex": float((o1 - o2).abs().max()),
                                        "max_abs_diff_final_gex": float((f1 - f2).abs().max()),
                                        "max_abs_final_change_published": float((f1 - o1).abs().max())}
    json.dump(res, open(os.path.join(out, "equivalence_check.json"), "w"), indent=1)
    print(json.dumps(res), flush=True)
    return res


def spearman(a, b):
    return NG.spearman(a, b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--lr", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--train-epochs", type=int, default=200)
    ap.add_argument("--sim-epochs", type=int, default=50)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--check-real", action="store_true", help="published vs vectorised SimComen on the real Tgfbr2 tissue")
    ap.add_argument("--kos", nargs="*", default=None)
    ap.add_argument("--max-candidates", type=int, default=0, help="timing test only")
    ap.add_argument("--evaluate-only", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    if a.check:
        check_equivalence(a.out)
        return
    t0 = time.time()
    D = prepare_data(a.npz, a.raw, a.lr)
    genes, readout, r_idx = D["genes"], D["readout"], D["r_idx"]
    ref_info = json.load(open(os.path.join(HERE, "..", "results", "perturbmap", "run_info.json")))
    info = {"n_genes_model": len(genes), "n_readout": len(readout), "n_train_spots": int(len(D["train"])),
            "n_candidates": {k: len(v) for k, v in D["candidates"].items()},
            "check_matches_paper_run_info": (len(genes) == ref_info["n_genes_model"]
                                             and len(readout) == ref_info["n_readout"]
                                             and int(len(D["train"])) == ref_info["n_train_spots"]
                                             and {k: len(v) for k, v in D["candidates"].items()} == ref_info["n_candidates"])}
    print(json.dumps(info), flush=True)

    import scanpy as sc
    import anndata
    from scipy import sparse
    from celcomen.models.celcomen import celcomen
    from celcomen.utils.helpers import normalize_g2g
    from celcomen.training_plan.train import train
    from celcomen.datareaders.datareader import get_dataset_loaders
    simcomen, Fast = make_fast_simcomen()

    train_idx = D["train"]
    pos_in_train = -np.ones(D["n_spots"], int)
    pos_in_train[train_idx] = np.arange(len(train_idx))
    # training graph as Celcomen builds it (k = 6 within each section), also used to pick 1-hop neighbours
    from sklearn.neighbors import kneighbors_graph
    sec = D["ann"].section.to_numpy()[train_idx]
    rows, cols = [], []
    for s in np.unique(sec):
        idx = np.where(sec == s)[0]
        A = kneighbors_graph(D["coords"][train_idx][idx], n_neighbors=K_SPOT, include_self=False).tocoo()
        rows.append(idx[A.row]); cols.append(idx[A.col])
    E = np.vstack([np.concatenate(rows), np.concatenate(cols)])

    if not a.evaluate_only:
        for seed in a.seeds:
            mpath = os.path.join(a.out, f"celcomen_seed{seed}.pt")
            if not os.path.exists(mpath):
                ad = anndata.AnnData(X=sparse.csr_matrix(D["X"][train_idx]), var=pd.DataFrame(index=genes))
                ad.obs["sample"] = sec
                ad.obsm["spatial"] = D["coords"][train_idx]
                h5 = os.path.join(a.out, "train_spots.h5ad")
                ad.write(h5)
                loader = get_dataset_loaders(h5, sample_id_name="sample", n_neighbors=K_SPOT, distance=None,
                                             device="cpu", verbose=False)
                n = len(genes)
                model = celcomen(input_dim=n, output_dim=n, n_neighbors=K_SPOT, seed=seed)
                np.random.seed(seed)
                g0 = np.random.uniform(size=(n, n)).astype("float32")
                g0 = normalize_g2g((g0 + g0.T) / 2)
                model.set_g2g(torch.from_numpy(g0))
                model.set_g2g_intra(torch.from_numpy(g0))
                tm = time.time()
                losses = train(a.train_epochs, 0.1, model, loader, zmft_scalar=0.1, seed=seed, device="cpu")
                torch.save({"g2g": model.conv1.lin.weight.detach().clone(),
                            "g2g_intra": model.lin.weight.detach().clone(), "losses": [float(x) for x in losses],
                            "train_s": time.time() - tm}, mpath)
                os.remove(h5)
                print(f"seed {seed} trained in {time.time() - tm:.0f}s; loss {losses[0]:.4g} -> {losses[-1]:.4g}",
                      flush=True)
            W = torch.load(mpath)
            for k in (a.kos or D["kos"]):
                epath = os.path.join(a.out, f"effects_seed{seed}_{k}.pkl")
                done = pickle.load(open(epath, "rb")) if os.path.exists(epath) else {}
                S = pos_in_train[D["cmp_spots"][k]]                                  # comparison spots (train idx)
                nb = np.unique(E[0][np.isin(E[1], S)])                              # senders into S
                nodes = np.union1d(S, nb)
                loc = {v: i for i, v in enumerate(nodes)}
                m_ = np.isin(E[0], nodes) & np.isin(E[1], nodes)
                ei = torch.from_numpy(np.vstack([[loc[v] for v in E[0][m_]], [loc[v] for v in E[1][m_]]]))
                Sl = np.array([loc[v] for v in S])
                Xs = D["X"][train_idx][nodes]
                if "base" not in done:
                    tb = time.time()
                    _, base = run_simcomen(Fast, W["g2g"], W["g2g_intra"], unit(Xs), ei, a.sim_epochs, 1e-3, 0.1, 0)
                    done["base"] = base[Sl].numpy()
                    done["meta"] = {"n_nodes": int(len(nodes)), "n_cmp": int(len(S)), "n_edges": int(m_.sum()),
                                    "base_s": time.time() - tb}
                    print(k, done["meta"], flush=True)
                if a.check_real:
                    # The published SimComen records ~G^2/2 in-place slice operations per step in the autograd graph
                    # and was killed by the OOM killer (> 12 GB) at G = 1,759 even for a 150-spot tissue, so the
                    # real-data comparison uses real expression and the trained Celcomen weights restricted to the
                    # first 40 spots and the first 300 model genes.
                    from celcomen.models.simcomen import simcomen as Pub
                    sub, gs = np.arange(40), np.arange(300)
                    msk = (ei[0] < 40) & (ei[1] < 40)
                    ei_s = ei[:, msk]
                    Wg, Wi = W["g2g"][gs][:, gs].contiguous(), W["g2g_intra"][gs][:, gs].contiguous()
                    tq = time.time()
                    _, f_pub = run_simcomen(Pub, Wg, Wi, unit(Xs[sub][:, gs]), ei_s, a.sim_epochs, 1e-3,
                                            0.1, 0, fast=False)
                    t_pub = time.time() - tq
                    tq = time.time()
                    _, f_fast = run_simcomen(Fast, Wg, Wi, unit(Xs[sub][:, gs]), ei_s, a.sim_epochs, 1e-3, 0.1, 0)
                    from celcomen.utils.helpers import calc_sphex as calc_sphex_published
                    xu = unit(Xs)
                    tq = time.time()
                    sp_pub = calc_sphex_published(xu)
                    t_sp_pub = time.time() - tq
                    tq = time.time()
                    sp_fast = calc_sphex_fast(xu)
                    t_sp_fast = time.time() - tq
                    same_nan = bool(torch.equal(torch.isnan(sp_pub), torch.isnan(sp_fast)))
                    fin = ~torch.isnan(sp_pub)
                    chk = {"calc_sphex_full_tissue": {"shape": list(xu.shape), "nan_pattern_identical": same_nan,
                                                      "bit_identical_finite": bool(torch.equal(sp_pub[fin], sp_fast[fin])),
                                                      "max_abs_diff_finite": float((sp_pub[fin] - sp_fast[fin]).abs().max()),
                                                      "n_nan": int((~fin).sum()),
                                                      "runtime_published_s": t_sp_pub, "runtime_fast_s": t_sp_fast},
                           "knockout": k, "epochs": a.sim_epochs, "n_nodes": 40, "n_genes": 300, "n_edges": int(msk.sum()),
                           "max_abs_diff_final_gex": float((f_pub - f_fast).abs().max()),
                           "max_abs_final_gex": float(f_pub.abs().max()),
                           "nan_published": int(torch.isnan(f_pub).sum()), "nan_fast": int(torch.isnan(f_fast).sum()),
                           "runtime_published_s": t_pub, "runtime_fast_s": time.time() - tq}
                    json.dump(chk, open(os.path.join(a.out, f"equivalence_check_real_{k}.json"), "w"), indent=1)
                    print(chk, flush=True)
                    return
                base = torch.from_numpy(done["base"])
                cands = D["candidates"][k][:a.max_candidates] if a.max_candidates else D["candidates"][k]
                tc = time.time()
                for c_i, j in enumerate(cands):
                    if j in done:
                        continue
                    Xp = Xs.copy()
                    Xp[Sl, j] = 0.0
                    _, fin = run_simcomen(Fast, W["g2g"], W["g2g_intra"], unit(Xp), ei, a.sim_epochs, 1e-3, 0.1, 0)
                    done[j] = (fin[Sl] - base).mean(0).numpy()[r_idx]
                    if c_i % 25 == 0:
                        pickle.dump(done, open(epath, "wb"))
                        print(f"{k} seed {seed}: {c_i + 1}/{len(cands)} candidates, {time.time() - tc:.0f}s", flush=True)
                pickle.dump(done, open(epath, "wb"))

    # ---- evaluation (same definitions as scripts/run_perturbmap.py and stats_supplement.py) ----------
    rows, readout_tab = [], {"gene": readout}
    rng = np.random.default_rng(0)
    boot_rows = []
    for k in D["kos"]:
        files = [os.path.join(a.out, f"effects_seed{s}_{k}.pkl") for s in a.seeds]
        effs = [pickle.load(open(f, "rb")) for f in files if os.path.exists(f)]
        cand = D["candidates"][k]
        if not effs or not all(all(j in e for j in cand) for e in effs):
            print(f"{k}: scan incomplete, not evaluated", flush=True)
            continue
        eff = {j: np.mean([e[j] for e in effs], 0) for j in cand}
        obs = D["observed"][k]
        true_j = D["gi"][KO_GENES[k]]
        sims = {j: spearman(eff[j], obs) for j in cand}
        sims = {j: (0.0 if np.isnan(v) else v) for j, v in sims.items()}
        order = sorted(sims, key=lambda j: -sims[j])
        rank = order.index(true_j) + 1 if true_j in sims else None
        ifng_r = [readout.index(x) for x in D["ifng"]]
        rows.append({"method": f"Celcomen/SimComen ({len(effs)} model{'s' if len(effs) > 1 else ''})", "knockout": k,
                     "n_ko_spots": int((D["ko"] == k).sum()), "spearman_pred_vs_obs": sims.get(true_j),
                     "rank_of_true_gene": rank, "n_candidates": len(cand),
                     "percentile": (1 - (rank - 1) / len(cand)) if rank else None,
                     "top5_genes": [str(genes[j]) for j in order[:5]],
                     "frac_ifng_genes_predicted_down": float((eff[true_j][ifng_r] < 0).mean()),
                     "mean_pred_ifng_change": float(eff[true_j][ifng_r].mean()),
                     "n_candidates_with_all_zero_effect": int(sum(np.all(eff[j] == 0) for j in cand))})
        readout_tab[f"observed_{k}"] = obs
        readout_tab[f"pred_Celcomen_{k}"] = eff[true_j]
        o, p = obs, eff[true_j]
        ok = ~(np.isnan(o) | np.isnan(p))
        o, p = o[ok], p[ok]
        sp = lambda u, w: pd.Series(u).rank().corr(pd.Series(w).rank())  # noqa: E731 (as stats_supplement.py)
        boot = [sp(o[i], p[i]) for i in (rng.integers(0, len(o), len(o)) for _ in range(2000))]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        boot_rows.append({"knockout": k, "method": rows[-1]["method"], "n_genes": len(o), "spearman": sp(o, p),
                          "ci95_low": lo, "ci95_high": hi})
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(a.out, "perturbmap_validation_celcomen.csv"), index=False)
        pd.DataFrame(boot_rows).to_csv(os.path.join(a.out, "spearman_bootstrap_ci_celcomen.csv"), index=False)
        pd.DataFrame(readout_tab).to_csv(os.path.join(a.out, "readout_effects_celcomen.csv"), index=False)
        print(pd.DataFrame(rows).to_string(), flush=True)
        print(pd.DataFrame(boot_rows).to_string(), flush=True)
    info["runtime_s"] = time.time() - t0
    json.dump(info, open(os.path.join(a.out, "run_info.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
