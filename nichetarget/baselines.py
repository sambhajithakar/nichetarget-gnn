"""Baseline target-ranking strategies used for comparison.

Each baseline has a global form and a niche-stratified form (computed within
each inferred niche and maximised over niches), so that baselines and
NicheTarget-GNN receive the same niche information.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

MIN_CELLS = 30


def de_score(X, target_cells, ref_cells):
    """Differential expression (target state minus reference state)."""
    return X[target_cells].mean(0) - X[ref_cells].mean(0)


def _neighbour_mean(X, coords, k):
    tree = cKDTree(coords)
    _, nbr = tree.query(coords, k=k + 1)
    return X[nbr].mean(1)


def _corr_columns(M, y):
    out = np.zeros(M.shape[1])
    for j in range(M.shape[1]):
        v = M[:, j]
        if v.std() < 1e-8:
            continue
        c = spearmanr(v, y).correlation
        out[j] = 0.0 if np.isnan(c) else c
    return out


def neighbourhood_correlation(X, coords, malignant, programme, k: int = 8, cells=None):
    """Spearman correlation between neighbourhood-mean expression and the state programme."""
    nb_mean = _neighbour_mean(X, coords, k)
    cells = np.where(malignant)[0] if cells is None else cells
    return _corr_columns(nb_mean[cells], programme[cells])


def lr_communication(X, coords, malignant, programme, genes, lr_pairs, k: int = 8, cells=None):
    """CellChat-like spatial LR scoring.

    For each LR pair, per-malignant-cell communication = sum over neighbours of
    ligand x own receptor; the pair score is its Spearman correlation with the
    state programme. Both genes of the pair receive the pair score.
    """
    gi = {g: i for i, g in enumerate(genes)}
    tree = cKDTree(coords)
    _, nbr = tree.query(coords, k=k + 1)
    nbr = nbr[:, 1:]
    cells = np.where(malignant)[0] if cells is None else cells
    out = np.zeros(X.shape[1])
    for l, r in lr_pairs:
        if l not in gi or r not in gi:
            continue
        comm = X[nbr, gi[l]].sum(1) * X[:, gi[r]]
        if comm[cells].std() < 1e-8:
            continue
        c = spearmanr(comm[cells], programme[cells]).correlation
        c = 0.0 if np.isnan(c) else c
        out[gi[l]] = max(out[gi[l]], c)
        out[gi[r]] = max(out[gi[r]], c)
    return out


def niche_stratified(score_fn, malignant_cells, niche, **kw):
    """Maximum of a per-cell-subset score over niches with enough malignant cells."""
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
