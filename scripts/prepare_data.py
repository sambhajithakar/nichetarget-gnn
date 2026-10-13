"""Convert the public downloads into the inputs used by run_xenium.py, run_perturbmap.py and evaluate_xenium.py.

    # Xenium: extracted Xenium Onboard Analysis outputs (cell_feature_matrix/, analysis/) + cells.csv.gz
    python scripts/prepare_data.py xenium --root xenium_rep1/x --cells xenium_rep1/rep1_cells.csv.gz --out rep1.pkl
    # Perturb-map (GEO GSE193460): one npz per filtered_feature_bc_matrix.h5
    python scripts/prepare_data.py perturbmap --raw perturbmap/raw --out perturbmap/npz
    # DepMap: per-gene breast summary from a CRISPRGeneEffect-style table (rows = models, columns = genes)
    python scripts/prepare_data.py depmap --effects combined_model_crispr_data.csv --genes panel_genes.txt --out panel_breast_gene_effect.csv
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nichetarget import h5lite, xenium  # noqa: E402


def prep_xenium(a):
    s = xenium.load_xenium(a.root, a.cells, min_counts=a.min_counts)
    pickle.dump({"X": xenium.lognorm(s.counts), "counts": s.counts, "genes": s.genes, "coords": s.coords,
                 "cluster": s.cluster, "cell_id": s.cell_id}, open(a.out, "wb"))
    print(f"{a.out}: {s.counts.shape[0]} cells x {s.counts.shape[1]} genes")


def prep_perturbmap(a):
    os.makedirs(a.out, exist_ok=True)
    for p in sorted(glob.glob(os.path.join(a.raw, "*_filtered_feature_bc_matrix.h5"))):
        m = h5lite.read_10x_h5(p)
        name = os.path.basename(p).replace("_filtered_feature_bc_matrix.h5", ".npz")
        np.savez(os.path.join(a.out, name), data=m["data"], indices=m["indices"], indptr=m["indptr"],
                 shape=np.array(m["shape"]), barcodes=np.array(m["barcodes"]),
                 feature_id=np.array(m["feature_id"]), feature_name=np.array(m["feature_name"]))
        print(f"{name}: {m['shape'][1]} spots x {m['shape'][0]} features")


def prep_depmap(a):
    meta = ["model_id", "cell_line_name", "stripped_cell_line_name", "oncotree_lineage", "oncotree_primary_disease",
            "oncotree_subtype"]
    genes = [g.strip() for g in open(a.genes) if g.strip()]
    df = pd.read_csv(a.effects, usecols=lambda c: c in meta or c in set(genes))
    breast = df[df.oncotree_lineage == "Breast"]
    rows = []
    for g in genes:
        if g not in df.columns:
            continue
        e = breast[g].dropna()
        rows.append({"gene": g, "mean_effect_breast": e.mean(), "frac_lines_dependent_breast": (e < -0.5).mean(),
                     "mean_effect_all": df[g].mean()})
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"{a.out}: {len(rows)} genes, {len(breast)} breast lines")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    x = sub.add_parser("xenium")
    x.add_argument("--root", required=True)
    x.add_argument("--cells", required=True)
    x.add_argument("--min-counts", type=int, default=10)
    x.add_argument("--out", required=True)
    p = sub.add_parser("perturbmap")
    p.add_argument("--raw", required=True)
    p.add_argument("--out", required=True)
    d = sub.add_parser("depmap")
    d.add_argument("--effects", required=True)
    d.add_argument("--genes", required=True)
    d.add_argument("--out", required=True)
    a = ap.parse_args()
    {"xenium": prep_xenium, "perturbmap": prep_perturbmap, "depmap": prep_depmap}[a.cmd](a)


if __name__ == "__main__":
    main()
