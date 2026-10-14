"""Print-ready (Elsevier) versions of Figs 1-5, Fig S1 and the graphical abstract.

Replaces the matplotlib figures of make_figures_static.py / summarise_simulation.py
(matplotlib is not available) with SVG layouts rendered to vector PDF + TIFF by
figvec.render(). All plotted values are read from the result files:

    results/simulation/all_results_*.json      -> Figs 3, 4, 5
    results/simulation_v2/all_results_*.json   -> Fig S1
    nichetarget.simulate.simulate_tissue(seed=1) -> Fig 2
    results/xenium_rep1/gene_scores.csv        -> graphical abstract (top-5 list)

Usage (from the repository root):
    python scripts/make_figures_vector.py OUT_DIR [--fig-csv-dir DIR] [--preview DIR]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import figvec as V  # noqa: E402
from figvec import text, line, rect, circle, marker, MM, W1, W2, OI  # noqa: E402

GENES = ["L1", "R1", "L2", "R2", "D1"]
HILITE = "#0072B2"
BASEGREY = "#B8B8B8"


# ----------------------------------------------------------------------------- data
def load(res_dir):
    """Same parsing as summarise_simulation.load (metrics part)."""
    rows = []
    for f in sorted(glob.glob(os.path.join(res_dir, "all_results_*.json"))):
        scen = os.path.basename(f).replace("all_results_", "").replace(".json", "")
        for r in json.load(open(f)):
            for m, v in r["metrics"].items():
                rows.append({"scenario": scen, "seed": r["seed"], "method": m, "AUPRC": v["auprc"],
                             **{f"rank_{g}": v["ranks"][g] for g in GENES}})
    return pd.DataFrame(rows)


def check_against_csv(df, csv):
    """Cross-check JSON-derived values against the figure CSV used for the old figures."""
    if not csv or not os.path.exists(csv):
        return
    c = pd.read_csv(csv)
    m = df.merge(c, on=["scenario", "seed", "method"], suffixes=("", "_csv"))
    assert len(m) == len(df) == len(c), (len(m), len(df), len(c))
    for col in ["AUPRC"] + [f"rank_{g}" for g in GENES]:
        assert np.allclose(m[col], m[col + "_csv"]), col
    print(f"  values match {os.path.basename(csv)} ({len(m)} rows)")


# ----------------------------------------------------------------------------- Fig 1
def rich(x, y, segs, size=7, anchor="middle", fill="#222", weight="normal"):
    """segs: list of (string, kind); kind in n (normal), i (italic), sub, isub, sup, isup."""
    out, shift = [], 0.0
    for s, k in segs:
        it = "i" in k
        fs = size * 0.72 if ("sub" in k or "sup" in k) else size
        target = size * 0.28 if "sub" in k else (-size * 0.38 if "sup" in k else 0.0)
        dy = target - shift
        shift = target
        out.append(f'<tspan dy="{dy:.2f}" font-size="{fs:.2f}"{" font-style=\"italic\"" if it else ""}>{s}</tspan>')
    return text(x, y, "".join(out), size, anchor, weight=weight, fill=fill, raw=True)


def fig1(out, prev):
    W, H = W2, 44 * MM
    n, gap, mx = 5, 13.0, 3.0
    bw = (W - 2 * mx - (n - 1) * gap) / n
    top, bh = 4.0, H - 8.0
    boxes = [
        ("1  Inputs", [[("Spatial expression", "n")], [("Cell coordinates", "n")], [("Cell states / types", "n")],
                       [("LR pairs (NicheNet,", "n")], [("CellPhoneDB)", "n")], [("Normal reference", "n")]]),
        ("2  Niche-aware graph", [[("k-NN cell graph", "n")], [("Edge: distance + LR", "n")],
                                  [("co-expression ", "n"), ("x", "i"), ("j,l", "isub"), (" ", "n"), ("x", "i"), ("i,r", "isub")],
                                  [("Niche descriptor ", "n"), ("ν", "i"), ("i", "isub")],
                                  [("(neighbourhood", "n")], [("composition)", "n")]]),
        ("3  Encoder", [[("Edge-aware graph", "n")], [("attention", "n")],
                        [("FiLM conditioning on ", "n"), ("ν", "i"), ("i", "isub")],
                        [("Read-out genes never", "n")], [("used as input", "n")],
                        [("Masked-expression", "n")], [("training", "n")]]),
        ("4  Counterfactual", [[("Inhibit gene ", "n"), ("g", "i"), (" (", "n"), ("x", "i"), ("g", "isub"), (" → 0)", "n")],
                               [("Recompute LR edges", "n")], [("Predict read-out shift", "n")],
                               [("Efficacy per niche ", "n"), ("E", "i"), ("g", "isub"), ("(", "sup"), ("n", "isup"), (")", "sup")]]),
        ("5  Ranking", [[("Niche-max efficacy", "n")], [("× selectivity ", "n"), ("σ", "i"), ("g", "isub")],
                        [("→ ranked,", "n")], [("niche-specific,", "n")], [("selective targets", "n")]]),
    ]
    body = []
    for i, (title, lines_) in enumerate(boxes):
        x = mx + i * (bw + gap)
        body.append(rect(x, top, bw, bh, "#EEF3FB", "#2A6FDB", 0.9, rx=5))
        body.append(text(x + bw / 2, top + 15, title, 8, "middle", weight="bold", fill="#1B3F80"))
        lh = 10.5
        y0 = top + 24 + (bh - 24) / 2 - (len(lines_) - 1) * lh / 2 + 2.5
        for j, segs in enumerate(lines_):
            body.append(rich(x + bw / 2, y0 + j * lh, segs, 7))
        if i < n - 1:
            body.append(V.arrow(x + bw + 1.5, top + bh / 2, x + bw + gap - 1.0, top + bh / 2, "#1B3F80", 1.0, 4.0))
    return V.render("\n".join(body), W, H, os.path.join(out, "Figure_1"), dpi=1000, preview_dir=prev)


# ----------------------------------------------------------------------------- Fig 2
def fig2(out, prev):
    from nichetarget.simulate import simulate_tissue, CELL_TYPES
    sim = simulate_tissue(seed=1)
    W = W2
    sq, top = 140.0, 18.0
    xs = [6.0, 182.0, 358.0]
    ct_label = {"malignant": "malignant", "macrophage": "macrophage", "fibroblast": "fibroblast",
                "tcell": "T cell", "endothelial": "endothelial"}
    ct_col = [OI["vermillion"], OI["orange"], OI["green"], OI["blue"], OI["grey"]]
    niche_names = ["background", "myeloid", "stromal", "immune"]
    niche_col = [OI["lightgrey"], OI["orange"], OI["green"], OI["blue"]]
    c = np.asarray(sim.coords, float)
    pad = 3.0

    def P(x0, xy):
        return x0 + pad + xy[:, 0] / 100 * (sq - 2 * pad), top + sq - pad - xy[:, 1] / 100 * (sq - 2 * pad)

    body = []
    titles = ["Cell types", "Planted niches", "Malignant-cell state"]
    for k, x0 in enumerate(xs):
        body.append(text(x0, 10, "ABC"[k], 9, weight="bold"))
        body.append(text(x0 + 10, 10, titles[k], 8))
        body.append(rect(x0, top, sq, sq, "none", "#555", 0.5))
    r = 0.8
    # A
    px, py = P(xs[0], c)
    for t in range(len(CELL_TYPES)):
        for i in np.where(sim.cell_type == t)[0]:
            body.append(circle(px[i], py[i], r, ct_col[t]))
    # B
    px, py = P(xs[1], c)
    for nid in range(4):
        for i in np.where(sim.niche == nid)[0]:
            body.append(circle(px[i], py[i], r, niche_col[nid]))
    # C
    mal = np.where(sim.cell_type == 0)[0]
    a = np.asarray(sim.aggressiveness, float)
    amin, amax = 0.0, 1.0
    cols = V.cmap((a[mal] - amin) / (amax - amin), V.MAGMA)
    px, py = P(xs[2], c)
    for i, col in zip(mal, cols):
        body.append(circle(px[i], py[i], r, col))
    # colour bar
    cbx, cbw = xs[2] + sq + 6, 6.0
    nseg = 100
    for s in range(nseg):
        v0 = s / nseg
        col = V.cmap([v0 + 0.5 / nseg], V.MAGMA)[0]
        y = top + sq - (s + 1) / nseg * sq
        body.append(rect(cbx, y, cbw, sq / nseg + 0.3, col))
    body.append(rect(cbx, top, cbw, sq, "none", "#555", 0.4))
    for v in (0, 0.2, 0.4, 0.6, 0.8, 1.0):
        y = top + sq - v * sq
        body.append(line(cbx + cbw, y, cbx + cbw + 2, y, "#555", 0.4))
        body.append(text(cbx + cbw + 3.5, y + 2.5, f"{v:.1f}", 7))
    body.append(rich(cbx + cbw + 24, top + sq / 2, [("aggressiveness ", "n"), ("a", "i"), ("i", "isub")], 7)
                .replace("<text ", f'<text transform="rotate(-90 {cbx + cbw + 24:.2f} {top + sq / 2:.2f})" ', 1))
    # legends
    ly = top + sq + 11
    body.append(V.legend_items(xs[0] + 2, ly, [(ct_label[t], ct_col[i]) for i, t in enumerate(CELL_TYPES)],
                               size=7, sw=5.5, col_w=56, ncol=3, row_h=10, kind="dot"))
    body.append(V.legend_items(xs[1] + 2, ly, list(zip(niche_names, niche_col)), size=7, sw=5.5, col_w=52, ncol=3,
                               row_h=10, kind="dot"))
    body.append(text(xs[2], ly, "Malignant cells only", 7, fill="#444"))
    body.append(text(xs[2], ly + 10, "Simulated field: 100 × 100 units (seed 1)", 7, fill="#444"))
    H = ly + 16
    return V.render("\n".join(body), W, H, os.path.join(out, "Figure_2"), dpi=1000, preview_dir=prev)


# ----------------------------------------------------------------------------- bar panels (Figs 3, 4, S1)
def bar_panel(d, order, x0, y0, pw, row_h, color_of, xlabel="AUPRC (selective targets)", title=None, letter=None):
    out = []
    n = len(order)
    ph = n * row_h
    X = lambda v: x0 + v * pw  # noqa: E731
    if letter:
        out.append(text(x0 - 2, y0 - 7, letter, 9, weight="bold"))
        out.append(text(x0 + 9, y0 - 7, title, 8))
    for v in np.arange(0, 1.01, 0.2):
        out.append(line(X(v), y0, X(v), y0 + ph, "#E4E4E4", 0.4))
    for i, m in enumerate(order):
        vals = d[d.method == m].AUPRC.to_numpy()
        mu, sd = vals.mean(), vals.std(ddof=1)
        yc = y0 + (i + 0.5) * row_h
        bh = row_h * 0.68
        out.append(rect(X(0), yc - bh / 2, mu * pw, bh, color_of(m)))
        lo, hi = max(mu - sd, 0), mu + sd
        out.append(line(X(lo), yc, X(hi), yc, "#000", 0.6))
        for e in (lo, hi):
            out.append(line(X(e), yc - 2.2, X(e), yc + 2.2, "#000", 0.6))
        for v in vals:
            out.append(circle(X(v), yc, 1.25, "#000", "white", 0.25))
    out.append(line(X(0), y0, X(0), y0 + ph, "#333", 0.6))
    out.append(line(X(0), y0 + ph, X(1), y0 + ph, "#333", 0.6))
    for v in np.arange(0, 1.01, 0.2):
        out.append(line(X(v), y0 + ph, X(v), y0 + ph + 2.5, "#333", 0.6))
        out.append(text(X(v), y0 + ph + 10, f"{v:.1f}", 7, "middle"))
    out.append(text(X(0.5), y0 + ph + 20, xlabel, 7.5, "middle"))
    return "\n".join(out), ph


MAIN = ["DE", "DE (niche-stratified)", "NeighbourhoodCorr", "NeighbourhoodCorr (niche-stratified)",
        "LR-communication", "LR-communication (niche-stratified)",
        "LR-communication (niche-stratified)+Selectivity", "NicheTarget-GNN",
        "NicheTarget-GNN (ensemble of 3)", "Hybrid (GNN ensemble + LR-communication)"]


DISPLAY = {"NeighbourhoodCorr": "Neighbourhood correlation",
           "NeighbourhoodCorr (niche-stratified)": "Neighbourhood correlation (niche-stratified)",
           "LR-communication (niche-stratified)+Selectivity": "LR-communication (niche-stratified) + selectivity"}


def disp(m):
    return DISPLAY.get(m, m)


def is_ours(m):
    return m.startswith(("NicheTarget-GNN", "Hybrid"))


def fig_benchmark(df, base, prev):
    W = W2
    lab_w, gap = 168.0, 22.0
    pw = (W - lab_w - gap - 12) / 2
    row_h = 15.0
    top = 18.0
    d_all = {s: df[(df.scenario == s) & (df.method.isin(MAIN))] for s in ["full-lr", "hidden-L1-R1"]}
    order = [m for m in MAIN if m in d_all["full-lr"].method.unique()]
    assert order == [m for m in MAIN if m in d_all["hidden-L1-R1"].method.unique()]
    body = []
    color_of = lambda m: HILITE if is_ours(m) else BASEGREY  # noqa: E731
    for k, (s, t) in enumerate([("full-lr", "Complete LR database"), ("hidden-L1-R1", "L1–R1 missing from LR database")]):
        x0 = lab_w + k * (pw + gap)
        svg, ph = bar_panel(d_all[s], order, x0, top, pw, row_h, color_of, title=t, letter="AB"[k])
        body.append(svg)
    for i, m in enumerate(order):
        body.append(text(lab_w - 5, top + (i + 0.5) * row_h + 2.5, disp(m), 7, "end",
                         weight="bold" if is_ours(m) else "normal"))
    H = top + len(order) * row_h + 26
    return V.render("\n".join(body), W, H, base, dpi=1000, preview_dir=prev)


def fig4(df, out, prev):
    d = df[df.scenario == "full-lr"]
    order = ["NicheTarget-GNN", "NicheTarget-GNN (no selectivity)", "NicheTarget-GNN (global efficacy)",
             "w/o spatial message passing", "w/o niche conditioning", "w/o LR edge features"]
    order = [m for m in order if m in d.method.unique()]
    W = W1
    lab_w = 122.0
    pw = W - lab_w - 10
    row_h, top = 16.0, 6.0
    svg, ph = bar_panel(d, order, lab_w, top, pw, row_h, lambda m: HILITE if m == "NicheTarget-GNN" else BASEGREY)
    body = [svg]
    for i, m in enumerate(order):
        body.append(text(lab_w - 5, top + (i + 0.5) * row_h + 2.5, m, 7, "end"))
    H = top + ph + 26
    return V.render("\n".join(body), W, H, os.path.join(out, "Figure_4"), dpi=1000, preview_dir=prev)


# ----------------------------------------------------------------------------- Fig 5 dot plot
def fig5(df, out, prev):
    methods = [("DE", "DE", OI["grey"], "square"),
               ("NeighbourhoodCorr", "Neighbourhood correlation", OI["orange"], "triangle"),
               ("LR-communication (niche-stratified)", "LR-communication (niche-stratified)", OI["green"], "diamond"),
               ("NicheTarget-GNN", "NicheTarget-GNN", OI["blue"], "circle")]
    d = df[df.scenario == "full-lr"]
    med = {(m, g): float(d[d.method == m][f"rank_{g}"].median()) for m, *_ in methods for g in GENES}
    W = W1
    lab_w, right = 26.0, 10.0
    pw = W - lab_w - right
    top, sub, grp_gap = 6.0, 6.0, 7.0
    lo, hi = np.log10(1), np.log10(200)
    X = lambda v: lab_w + (np.log10(v) - lo) / (hi - lo) * pw  # noqa: E731
    body = []
    y = top
    ycent = []
    for gi, g in enumerate(GENES):
        y0 = y
        h = len(methods) * sub
        if gi % 2 == 0:
            body.append(rect(lab_w, y0 - 2, pw, h + 4, "#F3F3F3"))
        for mi, (m, _, col, shp) in enumerate(methods):
            yc = y0 + (mi + 0.5) * sub
            body.append(line(X(1), yc, X(med[(m, g)]), yc, col, 0.5))
            body.append(marker(shp, X(med[(m, g)]), yc, 2.3, col, "white", 0.3))
        ycent.append(y0 + h / 2)
        body.append(text(lab_w - 5, y0 + h / 2 + 2.5, g, 7.5, "end", weight="bold"))
        y += h + grp_gap
    ph = y - grp_gap + 2 - top
    yb = top + ph
    for t in (1, 3, 10, 30, 100, 200):
        body.append(line(X(t), top - 2, X(t), yb, "#D0D0D0", 0.35, "1.5,1.5"))
        body.append(line(X(t), yb, X(t), yb + 2.5, "#333", 0.6))
        body.append(text(X(t), yb + 10, str(t), 7, "middle"))
    body.append(line(lab_w, yb, lab_w + pw, yb, "#333", 0.6))
    body.append(text(lab_w + pw / 2, yb + 20, "Median rank of planted target among 200 genes (log scale)", 7, "middle"))
    body.append(text(lab_w + pw / 2, yb + 29, "1 = ranked first; 5 seeds, complete LR database", 7, "middle", fill="#555"))
    ly = yb + 42
    items = [(lab, col) for _, lab, col, _ in methods]
    shapes = [s for *_, s in methods]
    body.append(V.legend_items(lab_w - 18, ly, items[:1], 7, 6, kind="mk", shapes=shapes[:1]))
    body.append(V.legend_items(lab_w + 40, ly, items[1:2], 7, 6, kind="mk", shapes=shapes[1:2]))
    body.append(V.legend_items(lab_w - 18, ly + 10, items[2:3], 7, 6, kind="mk", shapes=shapes[2:3]))
    body.append(V.legend_items(lab_w + 132, ly + 10, items[3:4], 7, 6, kind="mk", shapes=shapes[3:4]))
    H = ly + 16
    for (m, g), v in med.items():
        print(f"  Fig5 median rank {g:3s} {m:40s} {v:g}")
    return V.render("\n".join(body), W, H, os.path.join(out, "Figure_5"), dpi=1000, preview_dir=prev)


# ----------------------------------------------------------------------------- graphical abstract
def graphical_abstract(out, prev):
    g = pd.read_csv(os.path.join(ROOT, "results", "xenium_rep1", "gene_scores.csv"))
    g = g[~g.is_readout]
    col = [c for c in g.columns if c.startswith("NicheTarget-GNN (ensemble")][0]
    top5 = g.sort_values(col, ascending=False).head(5)[["gene", col]].to_numpy()
    W, H = 130 * MM, 60 * MM - 14
    body = [text(8, 13, "NicheTarget-GNN: niche-aware in-silico target prioritisation", 9.5, weight="bold", fill="#1B3F80"),
]
    n, gap, mx, top = 4, 10.0, 6.0, 22.0
    bw = (W - 2 * mx - (n - 1) * gap) / n
    bh = H - top - 8
    titles = [["Spatial", "cell graph"], ["Niche-aware", "GNN"], ["In-silico", "inhibition"], ["Niche-specific", "ranked targets"]]
    for i in range(n):
        x = mx + i * (bw + gap)
        body.append(rect(x, top, bw, bh, "#F4F7FC", "#2A6FDB", 0.8, rx=4))
        for j, t in enumerate(titles[i]):
            body.append(text(x + bw / 2, top + 11 + j * 9.5, t, 8, "middle", weight="bold", fill="#1B3F80"))
        if i < n - 1:
            body.append(V.arrow(x + bw + 1, top + bh / 2, x + bw + gap - 1, top + bh / 2, "#1B3F80", 1.0, 3.5))
    cy0 = top + 30
    # 1: hand-placed illustrative graph (not data)
    x = mx
    pts = np.array([[10, 8], [22, 5], [33, 12], [15, 20], [27, 24], [40, 26], [8, 33], [20, 38], [34, 40],
                    [45, 12], [47, 37], [13, 48], [28, 52], [42, 50]], float)
    cols = [OI["vermillion"], OI["vermillion"], OI["orange"], OI["vermillion"], OI["orange"], OI["vermillion"],
            OI["blue"], OI["vermillion"], OI["green"], OI["vermillion"], OI["green"], OI["blue"], OI["vermillion"],
            OI["green"]]
    sx = (bw - 52) / 2 + x
    for a in range(len(pts)):
        dd = np.linalg.norm(pts - pts[a], axis=1)
        for b in np.argsort(dd)[1:4]:
            if b > a or a not in np.argsort(np.linalg.norm(pts - pts[b], axis=1))[1:4]:
                body.append(line(sx + pts[a, 0], cy0 + pts[a, 1], sx + pts[b, 0], cy0 + pts[b, 1], "#9AA7B8", 0.5))
    for p, c in zip(pts, cols):
        body.append(circle(sx + p[0], cy0 + p[1], 2.6, c, "white", 0.4))
    body.append(text(x + bw / 2, top + bh - 6, "cells + LR edges", 7, "middle", fill="#444"))
    # 2: network icon
    x = mx + (bw + gap)
    layers = [3, 4, 4, 2]
    lx = [x + 14 + k * (bw - 28) / 3 for k in range(4)]
    node = {}
    for k, nn in enumerate(layers):
        for j in range(nn):
            node[(k, j)] = (lx[k], cy0 + 27 + (j - (nn - 1) / 2) * 12)
    for k in range(3):
        for j in range(layers[k]):
            for j2 in range(layers[k + 1]):
                (x1, y1), (x2, y2) = node[(k, j)], node[(k + 1, j2)]
                body.append(line(x1, y1, x2, y2, "#B5C3D6", 0.4))
    for (k, j), (px, py) in node.items():
        body.append(circle(px, py, 3.0, "#2A6FDB" if k in (1, 2) else "#7FA7E8", "white", 0.4))
    body.append(text(x + bw / 2, top + bh - 15, "LR-edge attention", 7, "middle", fill="#444"))
    body.append(text(x + bw / 2, top + bh - 6, "+ niche conditioning", 7, "middle", fill="#444"))
    # 3: inhibition
    x = mx + 2 * (bw + gap)
    gx, gyb = x + 16, cy0 + 40
    for k, (hb, hb2) in enumerate([(26, 26), (30, 0), (18, 18)]):
        bx = gx + k * 18
        body.append(rect(bx, gyb - hb, 9, hb, "#C9D3E2"))
        if k == 1:
            body.append(rect(bx, gyb - hb, 9, hb, "none", OI["vermillion"], 0.8))
            body.append(line(bx - 2, gyb - hb - 2, bx + 11, gyb + 2, OI["vermillion"], 1.1))
    body.append(line(gx - 3, gyb, gx + 48, gyb, "#555", 0.5))
    body.append(rich(gx + 22, gyb + 10, [("x", "i"), ("g", "isub"), (" → 0", "n")], 8))
    body.append(text(x + bw / 2, top + bh - 15, "predict read-out", 7, "middle", fill="#444"))
    body.append(text(x + bw / 2, top + bh - 6, "shift per niche", 7, "middle", fill="#444"))
    # 4: ranked list (data: Xenium Rep1, GNN ensemble, scaled niche-max efficacy)
    x = mx + 3 * (bw + gap)
    ly0 = cy0 + 2
    bmax = bw - 62
    for k, (gene, v) in enumerate(top5):
        yy = ly0 + k * 9.5
        body.append(text(x + 6, yy + 3, f"{k + 1}", 7, fill="#666"))
        body.append(text(x + 12, yy + 3, gene, 7))
        body.append(rect(x + 54, yy - 3, float(v) * bmax, 6, OI["vermillion"]))
    body.append(text(x + bw / 2, top + bh - 15, "Xenium breast Rep1", 7, "middle", fill="#444"))
    body.append(text(x + bw / 2, top + bh - 6, "(GNN ensemble, top 5)", 7, "middle", fill="#444"))
    print("  GA top5:", [(gname, round(float(v), 2)) for gname, v in top5])
    return V.render("\n".join(body), W, H, os.path.join(out, "Graphical_abstract"), dpi=600, preview_dir=prev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--fig-csv-dir", default=None, help="manuscript/figures (CSV cross-check)")
    ap.add_argument("--preview", default=None)
    ap.add_argument("--only", default=None, help="comma list: 1,2,3,4,5,S1,GA")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    only = set(a.only.split(",")) if a.only else None
    want = lambda k: only is None or k in only  # noqa: E731
    sizes = {}
    df = load(os.path.join(ROOT, "results", "simulation"))
    df2 = load(os.path.join(ROOT, "results", "simulation_v2"))
    if a.fig_csv_dir:
        check_against_csv(df, os.path.join(a.fig_csv_dir, "simulation_metrics_long.csv"))
        check_against_csv(df2, os.path.join(a.fig_csv_dir, "v2_simulation_metrics_long.csv"))
    if want("1"):
        sizes["Figure_1"] = fig1(a.out, a.preview)
    if want("2"):
        sizes["Figure_2"] = fig2(a.out, a.preview)
    if want("3"):
        sizes["Figure_3"] = fig_benchmark(df, os.path.join(a.out, "Figure_3"), a.preview)
    if want("4"):
        sizes["Figure_4"] = fig4(df, a.out, a.preview)
    if want("5"):
        sizes["Figure_5"] = fig5(df, a.out, a.preview)
    if want("S1"):
        sizes["Figure_S1"] = fig_benchmark(df2, os.path.join(a.out, "Figure_S1"), a.preview)
    if want("GA"):
        sizes["Graphical_abstract"] = graphical_abstract(a.out, a.preview)
    for k, v in sizes.items():
        print(f"{k}: {v[0]} x {v[1]} px")


if __name__ == "__main__":
    main()
