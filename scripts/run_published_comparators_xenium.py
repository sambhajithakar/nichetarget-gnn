"""Published comparators on the Xenium breast sections (Rep1, Rep2).

Step 1 ``prepare``  (per section): rebuilds exactly the cell states, read-out
programme, niches and LR list of scripts/run_xenium.py (functions imported from
that script; niches loaded from results/xenium_<rep>/niche.npy), checks that the
in-house LR-communication baseline is reproduced, then runs
  * COMMOT (zcang/COMMOT, collective optimal transport), and
  * LIANA+ bivariate (saezlab/liana-py; default local metric 'cosine'; local
    Moran's R as a post hoc sensitivity analysis, see simulation script),
on all cells with the same 35 panel LR pairs (lr_pairs_human.csv: NicheNet v2 +
CellPhoneDB v2.1.7 subunits + NicheNet v1 curated) that the in-house baseline uses.
Pair scores are converted to gene scores with the in-house LR-communication rule
(Spearman of the per-cell tool score with the programme across malignant
target+reference cells; both genes take the maximum; niche-stratified variant =
maximum over niches with >= 30 malignant cells). Spatial scale r = mean distance
to the 10th nearest neighbour (COMMOT dis_thr = r; LIANA+ gaussian bandwidth = r).
It also writes the inputs for NicheNet (receiver = invasive target-state cells).

Step 2 (R): scripts/run_published_comparators_nichenet.R

Step 3 ``evaluate``: builds a gene_scores table per section with the new methods
and evaluates them with scripts/evaluate_xenium.py::evaluate (identical metrics
and evidence sets), plus BH q-values.

    python scripts/run_published_comparators_xenium.py prepare --data rep1.pkl --section rep1 --run results/xenium_rep1 \
        --lr lr_pairs_human.csv --out results/published_comparators/xenium_rep1
    Rscript scripts/run_published_comparators_nichenet.R results/published_comparators/xenium_rep1 ligand_target_matrix_nsga2r_final.rds lr_network_human_21122021.rds
    python scripts/run_published_comparators_xenium.py evaluate --ot-assoc ... --ot-drugs ... --depmap ... --out results/published_comparators/xenium_evaluation
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from nichetarget import np_graph as NG, np_baselines as NB  # noqa: E402
import run_xenium as RX  # noqa: E402
from run_published_comparators_simulation import (pair_to_gene, corr_rule, run_commot, run_liana,  # noqa: E402
                                                  MIN_CELLS)


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


def prepare(a):
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    d = pickle.load(open(a.data, "rb"))
    X, genes, coords, cluster = d["X"], list(d["genes"]), d["coords"], d["cluster"]
    counts = d["counts"]
    del d
    gi = {g: i for i, g in enumerate(genes)}
    if a.section == "rep1":
        lineage_of, tgt_cl, ref_cl = RX.CLUSTER_MAP_REP1, RX.TARGET_CLUSTERS_REP1, RX.REF_CLUSTERS_REP1
    else:
        lineage_of = RX.marker_lineage(X, genes, cluster)
        tgt_cl, ref_cl = RX.tumour_states_by_reference(X, genes, cluster, lineage_of)
    cell_type = np.array([lineage_of.get(int(c), 10) for c in cluster])
    target_cells = np.where(np.isin(cluster, tgt_cl))[0]
    ref_cells = np.where(np.isin(cluster, ref_cl))[0]
    mal_cells = np.where(np.isin(cluster, tgt_cl + ref_cl))[0]
    malignant_states = np.zeros(len(X), bool)
    malignant_states[mal_cells] = True
    de = NB.de_score(X, target_cells, ref_cells)
    readout_idx = np.argsort(de)[::-1][:RX.N_READOUT]
    readout = [genes[i] for i in readout_idx]
    programme = X[:, readout_idx].mean(1)
    lr = pd.read_csv(a.lr)
    lr_all = sorted(set((l, r) for l, r in zip(lr.ligand, lr.receptor) if l in gi and r in gi and l != r))
    niche = np.load(os.path.join(a.run, "niche.npy"))
    ct_saved = np.load(os.path.join(a.run, "cell_type.npy"))
    info_ref = json.load(open(os.path.join(a.run, "run_info.json")))
    info = {"section": a.section, "n_cells": int(len(X)), "n_target": int(len(target_cells)),
            "n_reference": int(len(ref_cells)), "readout": readout, "n_lr_pairs": len(lr_all),
            "lr_pairs": [list(p) for p in lr_all],
            "check_readout_equal": readout == info_ref["readout"],
            "check_cell_type_equal": bool((ct_saved == cell_type).all()),
            "check_n_lr_equal": len(lr_all) == info_ref["n_lr_pairs_in_panel"]}

    # reproduction check of the in-house LR-communication (global) score
    _, nbr = NG.knn(coords, RX.K_GRAPH)
    lr_inhouse = NB.lr_communication(X, nbr, malignant_states, programme, genes, lr_all)
    ref_scores = pd.read_csv(os.path.join(a.run, "gene_scores.csv")).set_index("gene")
    diff = np.abs(ref_scores.loc[genes, "LR-communication"].to_numpy() - lr_inhouse)
    info["check_lr_communication_max_abs_diff"] = float(diff.max())
    dk, _ = NG.knn(coords, RX.K_GRAPH)
    r = float(dk[:, -1].mean())
    info["spatial_scale_r_um"] = r
    print(json.dumps(info), flush=True)

    scores = {}
    tc = time.time()
    pc = run_commot(X, coords, genes, lr_all, r)
    info["runtime_commot_s"] = time.time() - tc
    info["commot_pairs_scored"] = len(pc)
    g_, n_ = gene_scores_from_tool(pc, programme, genes, mal_cells, niche)
    scores["COMMOT"], scores["COMMOT (niche-stratified)"] = g_, n_
    scores["COMMOT native (total received signal, malignant)"] = pair_to_gene(
        {p: float(v[mal_cells].sum()) for p, v in pc.items()}, genes)
    pd.DataFrame({f"{l}-{r_}": v for (l, r_), v in pc.items()}).describe().T.to_csv(
        os.path.join(a.out, "commot_receiver_signal_summary.csv"))
    del pc
    print("COMMOT done", info["runtime_commot_s"], flush=True)

    tl = time.time()
    pcl, globl, _ = run_liana(X, coords, genes, lr_all, r, local_name="cosine")
    g_, n_ = gene_scores_from_tool(pcl, programme, genes, mal_cells, niche)
    scores["LIANA+ bivariate"], scores["LIANA+ bivariate (niche-stratified)"] = g_, n_
    scores["LIANA+ native (global Moran's R)"] = pair_to_gene(
        {p: (0.0 if np.isnan(v) else v) for p, v in globl.items()}, genes)
    info["liana_pairs_scored"] = len(pcl)
    del pcl
    pcm, _, _ = run_liana(X, coords, genes, lr_all, r, local_name="morans")
    g_, n_ = gene_scores_from_tool(pcm, programme, genes, mal_cells, niche)
    scores["LIANA+ bivariate, local Moran's R [post hoc]"] = g_
    scores["LIANA+ bivariate, local Moran's R (niche-stratified) [post hoc]"] = n_
    info["runtime_liana_s"] = time.time() - tl
    del pcm

    out = pd.DataFrame({"gene": genes, **scores})
    out["is_readout"] = out.gene.isin(readout)
    out.to_csv(os.path.join(a.out, "gene_scores_lr_tools.csv"), index=False)

    # ---- NicheNet inputs (receiver = invasive target-state cells) ---------------------------
    pct = pd.DataFrame({"gene": genes, "pct_receiver": (counts[target_cells] > 0).mean(0)})
    for t, name in enumerate(RX.LINEAGES):
        m = cell_type == t
        pct[f"pct_{name.replace(' ', '_')}"] = (counts[m] > 0).mean(0) if m.any() else 0.0
    pct["is_readout"] = pct.gene.isin(readout)
    pct.to_csv(os.path.join(a.out, "nichenet_inputs_pct_expressed.csv"), index=False)
    info["runtime_s"] = time.time() - t0
    json.dump(info, open(os.path.join(a.out, "run_info.json"), "w"), indent=1)


def evaluate(a):
    import evaluate_xenium as EX
    from stats_supplement import bh
    os.makedirs(a.out, exist_ok=True)
    ot = pd.read_csv(a.ot_assoc)
    ot_score = dict(zip(ot.symbol, ot.score))
    dr = pd.read_csv(a.ot_drugs)
    dm = pd.read_csv(a.depmap)
    evidence = {"OT_breast": set(ot.symbol[ot.score >= 0.3]),
                "OT_clinical": set(dr.symbol[dr.max_clinical_stage.notna()]),
                "DepMap_breast": set(dm.gene[dm.frac_lines_dependent_breast >= 0.3])}
    res_all, tables = [], {}
    for sec in a.sections:
        d = pd.read_csv(os.path.join(a.root, f"xenium_{sec}", "gene_scores_lr_tools.csv"))
        nn_path = os.path.join(a.root, f"xenium_{sec}", "nichenet_gene_scores.csv")
        if os.path.exists(nn_path):
            nn = pd.read_csv(nn_path).set_index("gene")
            for c in [c for c in nn.columns if c.startswith("NicheNet")]:
                d[c] = d.gene.map(nn[c]).fillna(0.0).to_numpy()
        res = EX.evaluate(d, evidence, ot_score)
        res.insert(0, "section", f"xenium_{sec}")
        res_all.append(res)
        tables[sec] = d
        d.to_csv(os.path.join(a.out, f"gene_scores_all_published_{sec}.csv"), index=False)
    ev = pd.concat(res_all)
    ev.to_csv(os.path.join(a.out, "evaluation_published_comparators.csv"), index=False)
    rows = [(r.section, r.method, s, r[f"{s}_top25"], r[f"{s}_top25_p"])
            for _, r in ev.iterrows() for s in ("OT_breast", "OT_clinical", "DepMap_breast")]
    q = pd.DataFrame(rows, columns=["section", "method", "evidence_set", "top25", "p"])
    q["q_bh_new_family"] = bh(q.p)
    # sensitivity: the 60 original tests plus these new tests in one family (original q-values are NOT changed)
    orig = pd.read_csv(a.orig_bh)
    allp = np.concatenate([orig.p.to_numpy(), q.p.to_numpy()])
    q["q_bh_extended_family"] = bh(allp)[len(orig):]
    q["n_tests_new_family"] = len(q)
    q["n_tests_extended_family"] = len(allp)
    q.to_csv(os.path.join(a.out, "evidence_tests_bh_published_comparators.csv"), index=False)
    # replicate concordance (same definition as evaluate_xenium.py)
    if len(tables) == 2:
        d1, d2 = tables[a.sections[0]], tables[a.sections[1]]
        common = sorted(set(d1.gene[~d1.is_readout]) & set(d2.gene[~d2.is_readout]))
        conc = []
        for m in EX.method_columns(d1):
            s1 = d1.set_index("gene").loc[common, m].to_numpy(float)
            s2 = d2.set_index("gene").loc[common, m].to_numpy(float)
            t1 = set(np.array(common)[np.argsort(-s1, kind="mergesort")[:25]])
            t2 = set(np.array(common)[np.argsort(-s2, kind="mergesort")[:25]])
            conc.append({"method": m, "n_common_candidates": len(common), "spearman_rep1_rep2": NG.spearman(s1, s2),
                         "top25_overlap": len(t1 & t2), "shared_top_genes": ";".join(sorted(t1 & t2))})
        pd.DataFrame(conc).to_csv(os.path.join(a.out, "replicate_concordance_published_comparators.csv"), index=False)
    pd.set_option("display.width", 250)
    print(ev.drop(columns=["top25"]).round(3).to_string())
    print(q.round(4).to_string())


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--data", required=True)
    p.add_argument("--section", choices=["rep1", "other"], required=True)
    p.add_argument("--run", required=True, help="existing in-house run directory (niche.npy, gene_scores.csv)")
    p.add_argument("--lr", required=True)
    p.add_argument("--out", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--root", default="results/published_comparators")
    e.add_argument("--sections", nargs="+", default=["rep1", "rep2"])
    e.add_argument("--ot-assoc", required=True)
    e.add_argument("--ot-drugs", required=True)
    e.add_argument("--depmap", required=True)
    e.add_argument("--orig-bh", default="results/xenium_evaluation/evidence_tests_bh.csv")
    e.add_argument("--out", default="results/published_comparators/xenium_evaluation")
    a = ap.parse_args()
    prepare(a) if a.cmd == "prepare" else evaluate(a)


if __name__ == "__main__":
    main()
