"""Restrict a prepared section (.pkl) to a gene list, re-normalising from raw counts.

Used for the discovery re-run on the 180 genes shared with the external 5K-panel cohort
(results/revision/PRESPECIFIED_REVISION.md).

    python scripts/subset_genes.py --data rep1.pkl --genes results/revision/G180_genes.txt --out rep1_g180.pkl
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nichetarget import xenium  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--genes", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = pickle.load(open(a.data, "rb"))
    keep = set(open(a.genes).read().split())
    idx = [i for i, g in enumerate(d["genes"]) if g in keep]
    counts = np.asarray(d["counts"])[:, idx].astype(np.float32)
    out = dict(d, counts=counts, X=xenium.lognorm(counts), genes=[d["genes"][i] for i in idx])
    pickle.dump(out, open(a.out, "wb"))
    print(f"{a.out}: {counts.shape[0]} cells x {counts.shape[1]} genes")


if __name__ == "__main__":
    main()
