"""Baselines without SciPy (mirrors ``baselines.py``)."""
from __future__ import annotations

import numpy as np

from .np_graph import knn, spearman_columns, spearman

MIN_CELLS = 30


def de_score(X, target_cells, ref_cells):
    return X[target_cells].mean(0) - X[ref_cells].mean(0)


def neighbour_index(coords, k):
    return knn(coords, k, include_self=False)[1]


def neighbourhood_correlation(X, nbr, malignant, programme, cells=None, **_):
    """Spearman correlation between neighbourhood-mean expression (cell + k neighbours) and the programme."""
    cells = np.where(malignant)[0] if cells is None else cells
    full = np.concatenate([cells[:, None], nbr[cells]], 1)
    nb_mean = X[full].mean(1)
    return spearman_columns(nb_mean, programme[cells])


def lr_communication(X, nbr, malignant, programme, genes, lr_pairs, cells=None, **_):
    gi = {g: i for i, g in enumerate(genes)}
    cells = np.where(malignant)[0] if cells is None else cells
    out = np.zeros(X.shape[1])
    for l, r in lr_pairs:
        if l not in gi or r not in gi:
            continue
        comm = X[nbr[cells], gi[l]].sum(1) * X[cells, gi[r]]
        if comm.std() < 1e-8:
            continue
        c = spearman(comm, programme[cells])
        c = 0.0 if np.isnan(c) else c
        out[gi[l]] = max(out[gi[l]], c)
        out[gi[r]] = max(out[gi[r]], c)
    return out


def niche_stratified(score_fn, malignant_cells, niche, **kw):
    best = None
    for n in np.unique(niche[malignant_cells]):
        cells = malignant_cells[niche[malignant_cells] == n]
        if len(cells) < MIN_CELLS:
            continue
        s = score_fn(cells=cells, **kw)
        best = s if best is None else np.maximum(best, s)
    return best


def de_niche_max(X, target_cells, ref_cells, niche):
    best = None
    for n in np.unique(niche[target_cells]):
        t = target_cells[niche[target_cells] == n]
        r = ref_cells[niche[ref_cells] == n]
        if len(t) < MIN_CELLS or len(r) < MIN_CELLS:
            continue
        s = de_score(X, t, r)
        best = s if best is None else np.maximum(best, s)
    return best
