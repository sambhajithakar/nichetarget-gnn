"""Fig. 6: NicheTarget-GNN on Xenium FFPE human breast cancer (Rep1).

    python scripts/make_fig_xenium.py --data rep1.pkl --run results/xenium_rep1 \
        --eval results/xenium_evaluation/evaluation_all.csv --out ../manuscript/submission_cbm/figures/Figure_6

Print version (Elsevier): laid out in points at 190 mm width (all text >= 7 pt),
written as vector PDF + TIFF by figvec.render(); the three spatial maps are
embedded rasters (~750 dpi at print size). A 1 mm scale bar is drawn under each
map (Xenium cell centroids are in microns).
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
import figlib as F  # noqa: E402  (raster_points)
import figvec as V  # noqa: E402
from figvec import text, line, rect, circle, W2  # noqa: E402

LIN_COL = {"malignant": "#D55E00", "myoepithelial": "#CC79A7", "fibroblast": "#009E73", "perivascular": "#8C564B",
           "endothelial": "#56B4E9", "macrophage": "#E69F00", "T cell": "#0072B2", "B cell": "#17BECF",
           "plasma cell": "#BCBD22", "mast cell": "#000000", "immune-stromal mixed": "#999999"}
NICHE_COL = ["#D55E00", "#999999", "#0072B2", "#009E73", "#CC79A7"]


def hbar(x0, y0, w, labels, values, color, label_w=42, bar_h=7.0, gap=2.6, xlabel=None):
    out = []
    pw = w - label_w - 20
    vmax = max(max(values), 1e-9)
    for i, (lab, v) in enumerate(zip(labels, values)):
        y = y0 + i * (bar_h + gap)
        out.append(text(x0 + label_w - 3, y + bar_h - 1.2, lab, 7, "end"))
        out.append(rect(x0 + label_w, y, max(v, 0) / vmax * pw, bar_h, color))
        out.append(text(x0 + label_w + max(v, 0) / vmax * pw + 2, y + bar_h - 1.2, f"{v:.2f}", 7))
    h = len(labels) * (bar_h + gap)
    if xlabel:
        out.append(text(x0 + label_w + pw / 2, y0 + h + 8, xlabel, 7, "middle", fill="#444"))
        h += 10
    return "\n".join(out), h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out", required=True, help="output base path (extension ignored)")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--preview", default=None)
    a = ap.parse_args()
    base = os.path.splitext(a.out)[0]
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
    span_x = float(coords[:, 0].max() - coords[:, 0].min())   # raster_points scales by the x span
    W = W2
    pw = 172.0
    xs = [4.0, 183.0, 362.0]
    RPX, PT = 1800, 1.3                 # raster width (px) and point radius (px)
    body = []
    href, ph_px = F.raster_points(coords[order], [LIN_COL[lineages[t]] for t in ct[order]], RPX, point=PT)
    mh = pw * ph_px / RPX
    my = 14.0
    titles = [f"Cell lineages (Rep1, {len(coords):,} cells)", "Malignant states", "Niches (k-means on neighbourhood)"]
    for k, x0 in enumerate(xs):
        body += [text(x0, 9, "ABC"[k], 9, weight="bold"), text(x0 + 10, 9, titles[k], 7.5)]
    body.append(F.image(href, xs[0], my, pw, mh))
    col = np.where(tgt, "#D55E00", np.where(ref, "#0072B2", "#DDDDDD"))
    o2 = np.argsort(tgt | ref, kind="stable")
    href, _ = F.raster_points(coords[o2], list(col[o2]), RPX, point=PT)
    body.append(F.image(href, xs[1], my, pw, mh))
    href, _ = F.raster_points(coords[order], [NICHE_COL[n] for n in niche[order]], RPX, point=PT)
    body.append(F.image(href, xs[2], my, pw, mh))
    # 1 mm scale bars (1000 um)
    bar = 1000.0 / span_x * pw
    sb_y = my + mh + 5
    for x0 in xs:
        xe = x0 + pw - 2
        body.append(line(xe - bar, sb_y, xe, sb_y, "#000", 1.4))
        body.append(text(xe - bar - 3, sb_y + 2.4, "1 mm", 7, "end"))
    ly = sb_y + 12
    body.append(V.legend_items(xs[0], ly, list(LIN_COL.items()), size=7, sw=5.5, col_w=86, ncol=2, row_h=9))
    body.append(V.legend_items(xs[1], ly, [(f"invasive tumour (target, n = {info['n_target']:,})", "#D55E00"),
                                           (f"DCIS (reference, n = {info['n_reference']:,})", "#0072B2"),
                                           ("other cells", "#DDDDDD")], size=7, sw=5.5, row_h=9))
    comp = info["niche_composition"]
    labs = []
    for k in sorted(comp, key=int):
        top = sorted(comp[k].items(), key=lambda kv: -kv[1])[:2]
        labs.append((f"N{k}: " + ", ".join(f"{n} {v:.0%}" for n, v in top), NICHE_COL[int(k)]))
    body.append(V.legend_items(xs[2], ly, labs, size=7, sw=5.5, row_h=9))
    y0 = ly + 6 * 9 + 14
    # D top targets of the GNN ensemble
    g = pd.read_csv(os.path.join(a.run, "gene_scores.csv"))
    g = g[~g.is_readout]
    hy = "Hybrid (GNN ensemble + LR-communication)"
    gnn = [c for c in g.columns if c.startswith("NicheTarget-GNN (ensemble")][0]
    top = g.sort_values(gnn, ascending=False).head(15)
    svg, h1 = hbar(xs[0], y0 + 8, pw, list(top.gene), list(top[gnn]), "#D55E00",
                   xlabel="niche-max efficacy (scaled to max)")
    body += [text(xs[0], y0, "D", 9, weight="bold"), text(xs[0] + 10, y0, "Top 15, NicheTarget-GNN ensemble", 7.5), svg]
    lr = "LR-communication (niche-stratified)"
    top2 = g.sort_values(lr, ascending=False).head(15)
    svg, _ = hbar(xs[1], y0 + 8, pw, list(top2.gene), list(top2[lr]), "#0072B2",
                  xlabel="Spearman with read-out (max over niches)")
    body += [text(xs[1], y0, "E", 9, weight="bold"),
             text(xs[1] + 10, y0, "Top 15, LR-communication (niche-stratified)", 7.5), svg]
    # F evidence: AUPRC / prevalence for four methods and three evidence sets
    ev = pd.read_csv(a.eval)          # evaluation_all.csv: bars = mean over sections, dots = sections
    meths = [("DE", "DE"), (lr, "LR-comm. (niche-strat.)"), (gnn, "GNN ensemble"), (hy, "Hybrid")]
    sets = [("OT_breast", ["Open Targets", "breast"]), ("OT_clinical", ["Clinical-stage", "drug"]),
            ("DepMap_breast", ["DepMap breast", "dependency"])]

    def enr(m, s):
        r = ev[ev.method == m]
        return (r[f"{s}_AUPRC"] / r[f"{s}_prevalence"]).to_numpy()

    maxv = max(enr(m, s).max() for m, _ in meths for s, _ in sets)
    ymax = float(np.ceil(maxv / 2) * 2)
    x0, yb, ph2, bw = xs[2] + 20, y0 + 12, 104.0, 9.0
    Y = lambda v: yb + ph2 - v / ymax * ph2  # noqa: E731
    body += [text(xs[2], y0, "F", 9, weight="bold"), text(xs[2] + 10, y0, "External evidence (Rep1 and Rep2)", 7.5)]
    mcol = ["#999999", "#0072B2", "#D55E00", "#009E73"]
    for v in np.arange(0, ymax + 1e-9, 2):
        body.append(line(x0, Y(v), x0 + 150, Y(v), "#E6E6E6", 0.4))
        body.append(line(x0 - 2.5, Y(v), x0, Y(v), "#333", 0.6))
        body.append(text(x0 - 4, Y(v) + 2.5, f"{v:g}", 7, "end"))
    for si, (s, sl) in enumerate(sets):
        gx = x0 + 6 + si * 49
        for mi, (m, _) in enumerate(meths):
            vs = enr(m, s)
            body.append(rect(gx + mi * bw, Y(vs.mean()), bw - 1.5, Y(0) - Y(vs.mean()), mcol[mi]))
            for v in vs:
                body.append(circle(gx + mi * bw + (bw - 1.5) / 2, Y(v), 1.2, "#222", "white", 0.25))
        for j, t in enumerate(sl):
            body.append(text(gx + 2 * bw - 0.75, yb + ph2 + 9 + j * 8, t, 7, "middle"))
    body.append(line(x0, Y(1), x0 + 150, Y(1), "#555", 0.6, "2,2"))
    body.append(line(x0 + 5, Y(ymax) + 7, x0 + 17, Y(ymax) + 7, "#555", 0.6, "2,2"))
    body.append(text(x0 + 20, Y(ymax) + 9.4, "random expectation (= 1)", 7, fill="#555"))
    body.append(line(x0, yb, x0, yb + ph2, "#333", 0.6))
    body.append(line(x0, yb + ph2, x0 + 150, yb + ph2, "#333", 0.6))
    body.append(text(x0 - 15, yb + ph2 / 2, "AUPRC / prevalence", 7, "middle", rotate=-90))
    body.append(V.legend_items(x0 - 12, yb + ph2 + 32, [(lab, c) for (_, lab), c in zip(meths, mcol)],
                               size=7, sw=5.5, col_w=82, ncol=2, row_h=9))
    H = max(y0 + 8 + h1, yb + ph2 + 32 + 12) + 4
    for m, lab in meths:
        print("  F", lab, {s: np.round(enr(m, s), 2).tolist() for s, _ in sets})
    size = V.render("\n".join(body), W, H, base, dpi=a.dpi, preview_dir=a.preview)
    print(f"{os.path.basename(base)}: {size[0]} x {size[1]} px; raster maps {RPX} px over {pw:.0f} pt "
          f"({RPX / (pw / 72):.0f} dpi); 1 mm bar = {bar:.2f} pt (x span {span_x:.0f} um)")


if __name__ == "__main__":
    main()
