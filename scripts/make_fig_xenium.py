"""Fig. 6: NicheTarget-GNN on Xenium FFPE human breast cancer (Rep1).

    python scripts/make_fig_xenium.py --data rep1.pkl --run results/xenium_rep1 \
        --eval results/xenium_evaluation/evaluation_xenium_rep1.csv --out ../manuscript/figures/Fig6_xenium.png
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import figlib as F  # noqa: E402

LIN_COL = {"malignant": "#D55E00", "myoepithelial": "#CC79A7", "fibroblast": "#009E73", "perivascular": "#8C564B",
           "endothelial": "#56B4E9", "macrophage": "#E69F00", "T cell": "#0072B2", "B cell": "#17BECF",
           "plasma cell": "#BCBD22", "mast cell": "#000000", "immune-stromal mixed": "#999999"}
NICHE_COL = ["#D55E00", "#999999", "#0072B2", "#009E73", "#CC79A7"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = pickle.load(open(a.data, "rb"))
    coords = d["coords"]          # microns; y grows downwards as in the H&E image
    info = json.load(open(os.path.join(a.run, "run_info.json")))
    ct = np.load(os.path.join(a.run, "cell_type.npy"))
    niche = np.load(os.path.join(a.run, "niche.npy"))
    lineages = list(LIN_COL)
    cluster = d["cluster"]
    tgt = np.isin(cluster, info["target_clusters"])
    ref = np.isin(cluster, info["reference_clusters"])
    rng = np.random.default_rng(0)
    order = rng.permutation(len(coords))
    W = 1000
    body = []
    pw = 315
    # A lineages
    href, ph = F.raster_points(coords[order], [LIN_COL[lineages[t]] for t in ct[order]], 1260, point=0.9)
    body += [F.text(10, 18, "A", 14, weight="bold"), F.text(30, 18, "Cell lineages (Rep1, 164,000 cells)", 11),
             F.image(href, 10, 26, pw, pw * ph / 1260)]
    # B states
    col = np.where(tgt, "#D55E00", np.where(ref, "#0072B2", "#DDDDDD"))
    o2 = np.argsort(tgt | ref, kind="stable")
    href, _ = F.raster_points(coords[o2], list(col[o2]), 1260, point=0.9)
    body += [F.text(340, 18, "B", 14, weight="bold"), F.text(360, 18, "Malignant states", 11),
             F.image(href, 340, 26, pw, pw * ph / 1260)]
    # C niches
    href, _ = F.raster_points(coords[order], [NICHE_COL[n] for n in niche[order]], 1260, point=0.9)
    body += [F.text(670, 18, "C", 14, weight="bold"), F.text(690, 18, "Niches (k-means on neighbourhood)", 11),
             F.image(href, 670, 26, pw, pw * ph / 1260)]
    ymap = 26 + pw * ph / 1260 + 8
    body.append(F.legend(10, ymap, [(k, v) for k, v in LIN_COL.items()], size=9, sw=8, col_w=105, ncol=3))
    body.append(F.legend(340, ymap, [(f"invasive tumour (target, n={info['n_target']:,})", "#D55E00"),
                                     (f"DCIS (reference, n={info['n_reference']:,})", "#0072B2"),
                                     ("other cells", "#DDDDDD")], size=9, sw=8))
    comp = info["niche_composition"]
    labs = []
    for k in sorted(comp, key=int):
        top = sorted(comp[k].items(), key=lambda kv: -kv[1])[:2]
        labs.append((f"N{k}: " + ", ".join(f"{n} {v:.0%}" for n, v in top), NICHE_COL[int(k)]))
    body.append(F.legend(670, ymap, labs, size=9, sw=8))
    y0 = ymap + 5 * 13 + 30
    # D top targets of the hybrid with component ranks
    g = pd.read_csv(os.path.join(a.run, "gene_scores.csv"))
    g = g[~g.is_readout]
    hy = "Hybrid (GNN ensemble + LR-communication)"
    gnn = [c for c in g.columns if c.startswith("NicheTarget-GNN (ensemble")][0]
    top = g.sort_values(gnn, ascending=False).head(15)
    svg, h1 = F.hbar(10, y0 + 22, 320, list(top.gene), list(top[gnn]), color="#D55E00", label_w=70,
                     xlabel="niche-max efficacy (scaled to max)")
    body += [F.text(10, y0 + 10, "D", 14, weight="bold"), F.text(30, y0 + 10, "Top 15, NicheTarget-GNN ensemble", 11), svg]
    lr = "LR-communication (niche-stratified)"
    top2 = g.sort_values(lr, ascending=False).head(15)
    svg, _ = F.hbar(340, y0 + 22, 320, list(top2.gene), list(top2[lr]), color="#0072B2", label_w=70,
                    xlabel="Spearman with read-out (max over niches)")
    body += [F.text(340, y0 + 10, "E", 14, weight="bold"), F.text(360, y0 + 10, "Top 15, LR-communication", 11), svg]
    # F evidence: AUPRC / prevalence for four methods and three evidence sets
    ev = pd.read_csv(a.eval)          # evaluation_all.csv: bars = mean over sections, dots = sections
    meths = [("DE", "DE"), (lr, "LR-comm."), (gnn, "GNN ens."), (hy, "Hybrid")]
    sets = [("OT_breast", "Open Targets breast"), ("OT_clinical", "Clinical-stage drug"), ("DepMap_breast", "DepMap breast dep.")]
    x0, yb, bw = 690, y0 + 40, 13
    def enr(m, s):
        r = ev[ev.method == m]
        return (r[f"{s}_AUPRC"] / r[f"{s}_prevalence"]).to_numpy()
    maxv = max(enr(m, s).max() for m, _ in meths for s, _ in sets)
    ph2 = 210
    body += [F.text(670, y0 + 10, "F", 14, weight="bold"), F.text(690, y0 + 10, "External evidence (Rep1 and Rep2)", 11)]
    mcol = ["#999999", "#0072B2", "#D55E00", "#009E73"]
    for si, (s, sl) in enumerate(sets):
        gx = x0 + 20 + si * 95
        for mi, (m, _) in enumerate(meths):
            vs = enr(m, s)
            hh = vs.mean() / maxv * ph2
            body.append(f'<rect x="{gx + mi * bw}" y="{yb + ph2 - hh:.1f}" width="{bw - 2}" height="{hh:.1f}" fill="{mcol[mi]}"/>')
            for v in vs:
                body.append(f'<circle cx="{gx + mi * bw + (bw - 2) / 2:.1f}" cy="{yb + ph2 - v / maxv * ph2:.1f}" r="1.8" fill="#222"/>')
        body.append(F.text(gx + 2 * bw, yb + ph2 + 13, sl, 8.5, "middle"))
    y1 = yb + ph2 - 1 / maxv * ph2
    body.append(f'<line x1="{x0 + 10}" y1="{y1:.1f}" x2="{x0 + 300}" y2="{y1:.1f}" stroke="#555" stroke-dasharray="3,3"/>')
    body.append(F.text(x0 + 300, y1 - 3, "random", 8.5, "end", fill="#555"))
    body.append(f'<line x1="{x0 + 10}" y1="{yb}" x2="{x0 + 10}" y2="{yb + ph2}" stroke="#888"/>')
    for v in range(0, int(maxv) + 1, 2 if maxv > 6 else 1):
        yy = yb + ph2 - v / maxv * ph2
        body.append(F.text(x0 + 6, yy + 3, str(v), 8.5, "end", fill="#555"))
    body.append(F.text(x0 - 12, yb + ph2 / 2, "AUPRC / prevalence", 9.5, "middle", rotate=-90))
    body.append(F.legend(x0 + 20, yb + ph2 + 24, [(lab, c) for (_, lab), c in zip(meths, mcol)], size=9, sw=8, col_w=75, ncol=4))
    H = max(y0 + 22 + h1, yb + ph2 + 45) + 10
    F.render("\n".join(body), W, H, a.out, svg_path=a.out.replace(".png", ".svg"))


if __name__ == "__main__":
    main()
