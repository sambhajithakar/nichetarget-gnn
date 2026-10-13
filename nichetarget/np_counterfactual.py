"""Counterfactual inhibition for the NumPy model (mirrors ``counterfactual.py``).

For large sections the forward passes are restricted to the read-out cells
plus their n_layers-hop message-passing neighbourhood, which gives exactly the
same predictions for the read-out cells as a full-graph pass.
"""
from __future__ import annotations

import numpy as np

from .np_graph import hops, subgraph
from .np_model import make_input

MIN_NICHE_CELLS = 30


def state_axis(x, readout_idx, target_cells, ref_cells):
    mu_t = x[target_cells][:, readout_idx].mean(0)
    mu_r = x[ref_cells][:, readout_idx].mean(0)
    u = mu_r - mu_t
    return u, float((u ** 2).sum())


def scan_targets(model, g, genes, target_cells, ref_cells, niche, alpha=1.0, candidates=None,
                 chunk=None, axis_cells=None):
    """Return [(gene, overall efficacy, {niche: efficacy})] for every candidate gene.

    ``chunk``: process target cells in groups of this size (each with its own
    2-hop neighbourhood) to bound memory on large sections.
    """
    n_layers = model.n_layers if model.use_spatial else 0
    # the state axis uses all target cells (axis_cells) even when efficacy is
    # estimated on a random subset of them (target_cells)
    u, norm2 = state_axis(g.x, g.readout_idx, target_cells if axis_cells is None else axis_cells, ref_cells)
    readout = set(g.readout_idx.tolist())
    idx = list(candidates) if candidates is not None else list(range(len(genes)))
    groups = [target_cells] if chunk is None else [target_cells[i:i + chunk] for i in range(0, len(target_cells), chunk)]
    shift = np.zeros((len(idx), len(target_cells)), dtype=np.float64)
    off = 0
    for cells in groups:
        nodes = hops(g, cells, n_layers) if n_layers else np.asarray(cells)
        sg = subgraph(g, nodes)
        pos = np.searchsorted(nodes, cells)
        x_f = make_input(sg.x, sg.readout_idx)
        base = model.predict(x_f, sg)[pos][:, sg.readout_idx]
        for a, j in enumerate(idx):
            if j in readout:
                continue
            x_cf = x_f.copy()
            x_cf[:, j] *= (1.0 - alpha)
            pred = model.predict(x_cf, sg)[pos][:, sg.readout_idx]
            shift[a, off:off + len(cells)] = ((pred - base) @ u) / norm2
        off += len(cells)
    rows = []
    tn = niche[target_cells]
    for a, j in enumerate(idx):
        if j in readout:
            rows.append((genes[j], 0.0, {}))
            continue
        per = {}
        for nn in np.unique(tn):
            sel = tn == nn
            if sel.sum() >= MIN_NICHE_CELLS:
                per[int(nn)] = float(shift[a, sel].mean())
        rows.append((genes[j], float(shift[a].mean()), per))
    return rows, shift


def niche_max(per_niche):
    return max(per_niche.values()) if per_niche else 0.0


def selectivity(X_tumour, type_tumour, X_normal, type_normal, n_types):
    G = X_tumour.shape[1]
    mt = np.zeros((n_types, G))
    mn = np.zeros((n_types, G))
    for t in range(n_types):
        if (type_tumour == t).any():
            mt[t] = X_tumour[type_tumour == t].mean(0)
        if (type_normal == t).any():
            mn[t] = X_normal[type_normal == t].mean(0)
    best = mt.argmax(0)
    diff = mt[best, np.arange(G)] - mn[best, np.arange(G)]
    return 1.0 - np.exp(-np.clip(diff, 0.0, None))


def combine_scores(efficacy, select, lam=1.0):
    eff = np.clip(efficacy, 0, None)
    eff = eff / (eff.max() + 1e-12)
    if select is None:
        return eff
    return eff * np.power(select, lam)
