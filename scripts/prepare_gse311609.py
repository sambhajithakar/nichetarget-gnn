"""Convert GSE311609 (SPLIT / CHUV breast Xenium, 280-gene panel) sections into run_xenium.py inputs.

GSE311609 provides cell_feature_matrix.h5 and cells.parquet per section but no Xenium graph-based
clustering, so clusters are computed here with a fixed, pre-specified rule applied to every section:
log-normalised counts -> per-gene z-score -> PCA (20 components) -> k-means (k = 20, 5 restarts, seed 0).
Lineages and tumour states are then assigned by run_xenium.py (--section other), exactly as for Rep2.

    python scripts/prepare_gse311609.py --h5 GSMxxx_cell_feature_matrix.h5 --cells GSMxxx_cells.parquet.gz --out Bx.pkl
"""
from __future__ import annotations

import argparse
import gzip
import io
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nichetarget import h5lite, xenium, np_graph as NG  # noqa: E402

N_PCS, N_CLUSTERS = 20, 20


def load_cells(path):
    raw = open(path, "rb").read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return pd.read_parquet(io.BytesIO(raw))


def cluster_cells(X, seed=0):
    Z = (X - X.mean(0)) / (X.std(0) + 1e-6)
    U, S, _ = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    pcs = U[:, :N_PCS] * S[:N_PCS]
    return NG.kmeans(pcs, N_CLUSTERS, n_init=5, seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--cells", required=True)
    ap.add_argument("--min-counts", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    m = h5lite.read_10x_h5(a.h5)
    nf, nb = m["shape"]
    M = np.zeros((nb, nf), np.float32)
    for j in range(nb):
        s, e = m["indptr"][j], m["indptr"][j + 1]
        M[j, m["indices"][s:e]] = m["data"][s:e]
    is_gene = np.array([t == "Gene Expression" for t in m["feature_type"]])
    genes = [n for n, g in zip(m["feature_name"], is_gene) if g]
    cells = load_cells(a.cells)
    cells["cell_id"] = cells["cell_id"].astype(str)
    cells = cells.set_index("cell_id").loc[[str(b) for b in m["barcodes"]]]
    counts = M[:, is_gene]
    keep = counts.sum(1) >= a.min_counts
    counts = counts[keep]
    X = xenium.lognorm(counts)
    cluster = cluster_cells(X)
    pickle.dump({"X": X, "counts": counts, "genes": genes,
                 "coords": cells[["x_centroid", "y_centroid"]].to_numpy()[keep],
                 "cluster": cluster, "cell_id": np.array(m["barcodes"])[keep]}, open(a.out, "wb"))
    print(f"{a.out}: {counts.shape[0]} cells x {counts.shape[1]} genes, {len(np.unique(cluster))} clusters")


if __name__ == "__main__":
    main()
