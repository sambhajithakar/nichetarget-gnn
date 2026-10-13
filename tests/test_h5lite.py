"""Validate the pure-Python 10x HDF5 reader against the per-spot annotation tables.

Run from the repo root:  python3 -I tests/test_h5lite.py
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from nichetarget.h5lite import read_10x_h5  # noqa: E402

RAW = "/home/user/data/perturbmap/raw"


def check_sample(h5):
    m = read_10x_h5(h5)
    nf, nb = m["shape"]
    data, indices, indptr = m["data"], m["indices"], m["indptr"]
    assert indptr[-1] == len(data) == len(indices), "indptr[-1] != len(data)"
    assert indptr[0] == 0 and np.all(np.diff(indptr) >= 0), "indptr not monotone"
    assert indices.min() >= 0 and indices.max() < nf, "row index out of range"
    assert np.all(data > 0), "explicit zeros / negative counts"
    assert len(m["barcodes"]) == nb and len(m["feature_id"]) == nf == len(m["feature_name"])

    col_sum = np.add.reduceat(data, indptr[:-1]) if len(data) else np.zeros(nb, np.int64)
    col_sum[np.diff(indptr) == 0] = 0
    col_nnz = np.diff(indptr)

    ann = pd.read_csv(h5.replace("_filtered_feature_bc_matrix.h5", "_spot_annotation.csv.gz"))
    pos = {b: i for i, b in enumerate(m["barcodes"])}
    present = ann["barcode"].map(pos)
    found = present.notna()
    idx = present[found].astype(int).to_numpy()
    a = ann[found]
    cnt_ok = (col_sum[idx] == a["nCount_Spatial"].to_numpy()).mean()
    feat_ok = (col_nnz[idx] == a["nFeature_Spatial"].to_numpy()).mean()
    name = os.path.basename(h5).replace("_filtered_feature_bc_matrix.h5", "")
    print(f"{name}: shape={m['shape']} nnz={len(data)} annot={len(ann)} "
          f"in_matrix={found.sum()} nCount_match={cnt_ok:.4f} nFeature_match={feat_ok:.4f}")
    assert found.all(), "annotated barcodes missing from matrix"
    assert cnt_ok == 1.0 and feat_ok == 1.0, "column sums / nnz do not match annotation"
    return m


def main():
    files = sorted(glob.glob(os.path.join(RAW, "*_filtered_feature_bc_matrix.h5")))
    assert len(files) == 4, files
    for h5 in files:
        check_sample(h5)
    print("all h5lite checks passed")


if __name__ == "__main__":
    main()
