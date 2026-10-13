"""Benjamini-Hochberg q-values for the Xenium evidence tests and bootstrap CIs for the Perturb-map Spearman values.

    python scripts/stats_supplement.py --results results
"""
import argparse
import os

import numpy as np
import pandas as pd


def spearman(a, b):
    return pd.Series(a).rank().corr(pd.Series(b).rank())


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        j = order[i]
        prev = min(prev, p[j] * n / (i + 1))
        q[j] = prev
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    ev = pd.read_csv(os.path.join(a.results, "xenium_evaluation", "evaluation_all.csv"))
    rows = [(r.section, r.method, s, r[f"{s}_top25"], r[f"{s}_top25_p"])
            for _, r in ev.iterrows() for s in ("OT_breast", "OT_clinical", "DepMap_breast")]
    d = pd.DataFrame(rows, columns=["section", "method", "evidence_set", "top25", "p"])
    d["q_bh_all60"] = bh(d.p)
    d.to_csv(os.path.join(a.results, "xenium_evaluation", "evidence_tests_bh.csv"), index=False)

    x = pd.read_csv(os.path.join(a.results, "perturbmap", "readout_effects.csv"))
    rng = np.random.default_rng(0)
    out = []
    for k in ("Tgfbr2", "Ifngr2"):
        for m in ("NicheTarget-GNN", "w/o spatial message passing", "Co-expression regression"):
            o, p = x[f"observed_{k}"].to_numpy(), x[f"pred_{m}_{k}"].to_numpy()
            ok = ~(np.isnan(o) | np.isnan(p))
            o, p = o[ok], p[ok]
            boot = [spearman(o[i], p[i]) for i in (rng.integers(0, len(o), len(o)) for _ in range(a.n_boot))]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            out.append({"knockout": k, "method": m, "n_genes": len(o), "spearman": spearman(o, p),
                        "ci95_low": lo, "ci95_high": hi})
    pd.DataFrame(out).to_csv(os.path.join(a.results, "perturbmap", "spearman_bootstrap_ci.csv"), index=False)
    print(d.round(4).to_string())
    print(pd.DataFrame(out).round(3).to_string())


if __name__ == "__main__":
    main()
