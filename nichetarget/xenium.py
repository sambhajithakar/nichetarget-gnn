"""Loader for 10x Xenium Onboard Analysis outputs (no h5py/scanpy needed).

Reads ``cell_feature_matrix/`` (Matrix Market), ``cells.csv.gz`` and the
graph-based clustering from ``analysis/``. Returns gene-panel counts only
(negative-control probes and blank codewords are dropped, but their per-cell
totals are kept for QC).
"""
from __future__ import annotations

import gzip
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class XeniumSection:
    counts: np.ndarray        # cells x genes (float32), gene-panel counts
    genes: list[str]
    cell_id: np.ndarray
    coords: np.ndarray        # cells x 2 (microns)
    cells: pd.DataFrame       # cells.csv metadata, aligned
    cluster: np.ndarray       # graph-based cluster label (Xenium analysis)
    control_counts: np.ndarray


def read_mex(folder: str):
    feats = pd.read_csv(os.path.join(folder, "features.tsv.gz"), sep="\t", header=None)
    bcs = pd.read_csv(os.path.join(folder, "barcodes.tsv.gz"), sep="\t", header=None)[0].to_numpy()
    with gzip.open(os.path.join(folder, "matrix.mtx.gz"), "rt") as f:
        line = f.readline()
        while line.startswith("%"):
            line = f.readline()
        n_feat, n_bc, nnz = map(int, line.split())
        arr = pd.read_csv(f, sep=" ", header=None, dtype=np.int64).to_numpy()
    M = np.zeros((n_bc, n_feat), dtype=np.float32)
    M[arr[:, 1] - 1, arr[:, 0] - 1] = arr[:, 2]
    return M, feats, bcs


def load_xenium(root: str, cells_csv: str, min_counts: int = 10) -> XeniumSection:
    M, feats, bcs = read_mex(os.path.join(root, "cell_feature_matrix"))
    is_gene = (feats[2] == "Gene Expression").to_numpy()
    cells = pd.read_csv(cells_csv)
    cells["cell_id"] = cells["cell_id"].astype(str)
    cl = pd.read_csv(os.path.join(root, "analysis/clustering/gene_expression_graphclust/clusters.csv"))
    cl["Barcode"] = cl["Barcode"].astype(str)
    bcs = bcs.astype(str)
    cells = cells.set_index("cell_id").loc[bcs]
    clus = cl.set_index("Barcode")["Cluster"].reindex(bcs).fillna(-1).astype(int).to_numpy()
    counts = M[:, is_gene]
    keep = counts.sum(1) >= min_counts
    return XeniumSection(
        counts=counts[keep], genes=feats.loc[is_gene, 1].tolist(), cell_id=bcs[keep],
        coords=cells[["x_centroid", "y_centroid"]].to_numpy()[keep], cells=cells.iloc[np.where(keep)[0]].reset_index(),
        cluster=clus[keep], control_counts=M[:, ~is_gene].sum(1)[keep])


def lognorm(counts: np.ndarray, target: float | None = None) -> np.ndarray:
    """Library-size normalisation to the median total count, then log1p."""
    tot = counts.sum(1, keepdims=True)
    target = float(np.median(tot)) if target is None else target
    return np.log1p(counts / np.maximum(tot, 1) * target).astype(np.float32)
