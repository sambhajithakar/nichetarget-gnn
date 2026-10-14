"""Fig. 7: in-silico knockout vs. observed CRISPR-knockout lesion phenotypes (Perturb-map).

    python scripts/make_fig_perturbmap.py --run results/perturbmap --out ../manuscript/submission_cbm/figures/Figure_7

Print version (Elsevier): laid out in points at 190 mm width (all text >= 7 pt),
round axis ticks, Greek letters in the legend; written as vector PDF + TIFF by
figvec.render().
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import figvec as V  # noqa: E402
from figvec import text, line, rect, circle, num, W2  # noqa: E402

METHODS = [("NicheTarget-GNN", "NicheTarget-GNN", "#D55E00"),
           ("w/o spatial message passing", "Non-spatial variant", "#E69F00"),
           ("Co-expression regression", "Co-expression regression", "#999999")]
IFNG_COL, TGFB_COL, OTHER_COL = "#0072B2", "#009E73", "#C4C4C4"


def scatter(x0, y0, w, h, xs, ys, cols, labels, label_idx, xlabel, ylabel):
    """Scatter with round ticks; plot area starts at x0 + 30."""
    out = []
    px0, px1, py0, py1 = x0 + 30, x0 + w, y0, y0 + h
    xt = V.nice_ticks(xs.min(), xs.max(), 5)
    yt = V.nice_ticks(ys.min(), ys.max(), 6)
    xlo, xhi = min(xt[0], xs.min()), max(xt[-1], xs.max())
    ylo, yhi = min(yt[0], ys.min()), max(yt[-1], ys.max())
    dx, dy = (xhi - xlo) * 0.04, (yhi - ylo) * 0.04
    xlo, xhi, ylo, yhi = xlo - dx, xhi + dx, ylo - dy, yhi + dy
    X = lambda v: px0 + (v - xlo) / (xhi - xlo) * (px1 - px0)  # noqa: E731
    Y = lambda v: py1 - (v - ylo) / (yhi - ylo) * (py1 - py0)  # noqa: E731
    out.append(rect(px0, py0, px1 - px0, py1 - py0, "none", "#555", 0.6))
    out.append(line(X(0), py0, X(0), py1, "#CFCFCF", 0.5))
    out.append(line(px0, Y(0), px1, Y(0), "#CFCFCF", 0.5))
    # grey points first, coloured on top
    idx = sorted(range(len(xs)), key=lambda i: cols[i] != OTHER_COL)
    for i in idx:
        out.append(circle(X(xs[i]), Y(ys[i]), 1.35, cols[i], opacity=0.75 if cols[i] == OTHER_COL else 0.9))
    for i in label_idx:
        right = X(xs[i]) > px0 + 0.72 * (px1 - px0)
        below = Y(ys[i]) < py0 + 9
        out.append(text(X(xs[i]) + (-3 if right else 3), Y(ys[i]) + (7.5 if below else -2.5), labels[i], 7,
                        "end" if right else "start", weight="bold"))
    for v in xt:
        out.append(line(X(v), py1, X(v), py1 + 2.5, "#333", 0.6))
        out.append(text(X(v), py1 + 10, num(v), 7, "middle"))
    for v in yt:
        out.append(line(px0 - 2.5, Y(v), px0, Y(v), "#333", 0.6))
        out.append(text(px0 - 4, Y(v) + 2.5, num(v), 7, "end"))
    out.append(text((px0 + px1) / 2, py1 + 20, xlabel, 7, "middle"))
    out.append(text(x0 + 6, (py0 + py1) / 2, ylabel, 7, "middle", rotate=-90))
    return "\n".join(out), xt, yt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True, help="output base path (extension ignored)")
    ap.add_argument("--dpi", type=int, default=1000)
    ap.add_argument("--preview", default=None)
    a = ap.parse_args()
    base = os.path.splitext(a.out)[0]
    info = json.load(open(os.path.join(a.run, "run_info.json")))
    res = pd.read_csv(os.path.join(a.run, "perturbmap_validation.csv"))
    eff = pd.read_csv(os.path.join(a.run, "readout_effects.csv"))
    kos = list(dict.fromkeys(res.knockout))
    ifng, tgfb = set(info.get("ifng_genes_in_readout", [])), set(info.get("tgfb_genes_in_readout", []))
    col = [IFNG_COL if gname in ifng else TGFB_COL if gname in tgfb else OTHER_COL for gname in eff.gene]
    body, W = [], W2
    pw, ph, top = 168.0, 128.0, 26.0
    for i, k in enumerate(kos):
        x0 = 4 + i * 179
        r = res[(res.method == "NicheTarget-GNN") & (res.knockout == k)].iloc[0]
        obs, pred = eff[f"observed_{k}"].to_numpy(), eff[f"pred_NicheTarget-GNN_{k}"].to_numpy() * 1000
        lab = list(eff.gene)
        topi = [int(np.argmax(obs)), int(np.argmin(obs)), int(np.argmax(np.abs(pred)))]
        body.append(text(x0, 9, "AB"[i], 9, weight="bold"))
        body.append(text(x0 + 10, 9, f"{k} knockout (n = {int(r.n_ko_spots)} spots)", 7.5))
        body.append(text(x0 + 10, 18.5, f"Spearman ρ = {num(r.spearman_pred_vs_obs, '{:.2f}')}", 7.5))
        svg, xt, yt = scatter(x0, top, pw, ph, obs, pred, col, lab, topi,
                              "observed change (KO − comparison lesions)",
                              "predicted change (in-silico KO, ×0.001)")
        body.append(svg)
        print(f"  {k}: x ticks {xt}; y ticks {yt}; rho = {r.spearman_pred_vs_obs:.3f}; labelled {[lab[j] for j in topi]}")
    ly = top + ph + 34
    body.append(V.legend_items(10, ly, [("IFN-γ response genes", IFNG_COL), ("TGF-β response genes", TGFB_COL),
                                        ("other read-out genes", OTHER_COL)], size=7, sw=5.5, col_w=110, ncol=3,
                               kind="dot"))
    # C: percentile of the true knockout gene among candidates
    x0, yb, phc = 392.0, top, ph
    body += [text(362, 9, "C", 9, weight="bold"), text(372, 9, "Rank of the true knockout gene", 7.5)]
    bw = 13.0
    Y = lambda v: yb + phc - v * phc  # noqa: E731
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        body.append(line(x0, Y(v), x0 + 140, Y(v), "#EBEBEB", 0.4))
        body.append(line(x0 - 2.5, Y(v), x0, Y(v), "#333", 0.6))
        body.append(text(x0 - 4, Y(v) + 2.5, f"{v:.2f}", 7, "end"))
    for ki, k in enumerate(kos):
        gx = x0 + 12 + ki * 70
        for mi, (m, _, c) in enumerate(METHODS):
            rr = res[(res.method == m) & (res.knockout == k)]
            if rr.empty or pd.isna(rr.percentile.iloc[0]):
                continue
            v = float(rr.percentile.iloc[0])
            body.append(rect(gx + mi * bw, Y(v), bw - 2, Y(0) - Y(v), c))
            body.append(text(gx + mi * bw + (bw - 2) / 2, Y(v) - 2.5, f"{int(rr.rank_of_true_gene.iloc[0])}", 7, "middle"))
        n = int(res[res.knockout == k].n_candidates.iloc[0])
        body.append(text(gx + 1.5 * bw - 1, yb + phc + 10, k, 7, "middle", italic=False))
        body.append(text(gx + 1.5 * bw - 1, yb + phc + 18, f"({n:,} candidates)", 7, "middle"))
    body.append(line(x0, Y(0.5), x0 + 140, Y(0.5), "#555", 0.6, "2,2"))
    body.append(text(x0 + 140, Y(0.5) - 2.5, "random", 7, "end", fill="#555"))
    body.append(line(x0, yb, x0, yb + phc, "#333", 0.6))
    body.append(line(x0, yb + phc, x0 + 140, yb + phc, "#333", 0.6))
    body.append(text(x0 - 22, yb + phc / 2, "percentile (1 = ranked first)", 7, "middle", rotate=-90))
    body.append(text(372, 18.5, "numbers above bars: rank among candidates", 7, fill="#555"))
    for (_, lab, c), (lx, dy) in zip(METHODS, ((x0 - 32, 0), (x0 + 40, 0), (x0 - 32, 9))):
        body.append(V.legend_items(lx, ly + dy, [(lab, c)], size=7, sw=5.5))
    H = ly + 15
    size = V.render("\n".join(body), W, H, base, dpi=a.dpi, preview_dir=a.preview)
    print(f"{os.path.basename(base)}: {size[0]} x {size[1]} px")


if __name__ == "__main__":
    main()
