"""Fig. 7: in-silico knockout vs. observed CRISPR-knockout lesion phenotypes (Perturb-map).

    python scripts/make_fig_perturbmap.py --run results/perturbmap --out ../manuscript/figures/Fig7_perturbmap.png
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import figlib as F  # noqa: E402

METHODS = [("NicheTarget-GNN", "NicheTarget-GNN", "#D55E00"),
           ("w/o spatial message passing", "w/o spatial", "#E69F00"),
           ("Co-expression regression", "Co-expression", "#999999")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    info = json.load(open(os.path.join(a.run, "run_info.json")))
    res = pd.read_csv(os.path.join(a.run, "perturbmap_validation.csv"))
    eff = pd.read_csv(os.path.join(a.run, "readout_effects.csv"))
    kos = list(dict.fromkeys(res.knockout))
    ifng, tgfb = set(info.get("ifng_genes_in_readout", [])), set(info.get("tgfb_genes_in_readout", []))
    col = ["#0072B2" if gname in ifng else "#009E73" if gname in tgfb else "#BBBBBB" for gname in eff.gene]
    body, W = [], 1000
    pw, ph = 300, 270
    for i, k in enumerate(kos):
        x0 = 10 + i * (pw + 30)
        r = res[(res.method == "NicheTarget-GNN") & (res.knockout == k)].iloc[0]
        obs, pred = eff[f"observed_{k}"].to_numpy(), eff[f"pred_NicheTarget-GNN_{k}"].to_numpy() * 1000
        lab = list(eff.gene)
        top = [int(np.argmax(obs)), int(np.argmin(obs)), int(np.argmax(np.abs(pred)))]
        body.append(F.text(x0, 18, "AB"[i], 14, weight="bold"))
        body.append(F.text(x0 + 20, 18, f"{k} knockout (n = {int(r.n_ko_spots)} spots); Spearman = {r.spearman_pred_vs_obs:.2f}", 11))
        body.append(F.scatter(x0, 26, pw, ph, obs, pred, color=col, labels=lab, label_idx=top,
                              xlabel="observed change (KO - comparison lesions)", ylabel="predicted change (in-silico KO, x 0.001)"))
    body.append(F.legend(20, ph + 48, [("IFN-gamma response genes", "#0072B2"), ("TGF-beta response genes", "#009E73"),
                                       ("other read-out genes", "#BBBBBB")], size=9, sw=8, col_w=170, ncol=3))
    # C: percentile of the true knockout gene among candidates
    x0, yb, phc = 680, 50, 230
    body += [F.text(x0 - 10, 18, "C", 14, weight="bold"), F.text(x0 + 10, 18, "Rank of the true knockout gene", 11)]
    bw = 22
    for ki, k in enumerate(kos):
        gx = x0 + 30 + ki * 130
        for mi, (m, _, c) in enumerate(METHODS):
            rr = res[(res.method == m) & (res.knockout == k)]
            if rr.empty or pd.isna(rr.percentile.iloc[0]):
                continue
            v = float(rr.percentile.iloc[0])
            hh = v * phc
            body.append(f'<rect x="{gx + mi * bw}" y="{yb + phc - hh:.1f}" width="{bw - 3}" height="{hh:.1f}" fill="{c}"/>')
            body.append(F.text(gx + mi * bw + (bw - 3) / 2, yb + phc - hh - 3, f"{int(rr.rank_of_true_gene.iloc[0])}", 8, "middle"))
        n = int(res[res.knockout == k].n_candidates.iloc[0])
        body.append(F.text(gx + 1.5 * bw, yb + phc + 13, f"{k} ({n} candidates)", 9, "middle"))
    y5 = yb + phc - 0.5 * phc
    body.append(f'<line x1="{x0 + 20}" y1="{y5:.1f}" x2="{x0 + 300}" y2="{y5:.1f}" stroke="#555" stroke-dasharray="3,3"/>')
    body.append(F.text(x0 + 300, y5 - 3, "random", 8.5, "end", fill="#555"))
    body.append(f'<line x1="{x0 + 20}" y1="{yb}" x2="{x0 + 20}" y2="{yb + phc}" stroke="#888"/>')
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        body.append(F.text(x0 + 16, yb + phc - v * phc + 3, f"{v:.2f}", 8.5, "end", fill="#555"))
    body.append(F.text(x0 - 22, yb + phc / 2, "percentile (1 = ranked first)", 9.5, "middle", rotate=-90))
    body.append(F.text(x0 + 160, yb + phc + 28, "numbers above bars: rank", 8.5, "middle", fill="#555"))
    body.append(F.legend(x0 + 20, yb + phc + 38, [(lab, c) for _, lab, c in METHODS], size=9, sw=8, col_w=100, ncol=3))
    F.render("\n".join(body), W, ph + 70, a.out, svg_path=a.out.replace(".png", ".svg"))


if __name__ == "__main__":
    main()
