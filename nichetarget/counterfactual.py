"""Counterfactual target inhibition and target scoring.

For a candidate gene g we create an inhibited input in which g is scaled by
(1 - alpha) in every cell (a systemic drug), recompute ligand-receptor edge
features from the inhibited profile, and predict expression with the trained
model. Efficacy is the fraction of the target-to-reference state gap that is
closed in malignant cells of the target state, measured on the read-out genes
only (g itself is never part of the read-out, because read-out genes are not
model inputs). Efficacy is reported per niche and overall.
"""
from __future__ import annotations

import numpy as np
import torch

from .model import make_input

MIN_NICHE_CELLS = 30


@torch.no_grad()
def state_axis(x: torch.Tensor, readout_idx, target_cells, ref_cells):
    mu_t = x[target_cells][:, readout_idx].mean(0)
    mu_r = x[ref_cells][:, readout_idx].mean(0)
    u = mu_r - mu_t
    return u, (u ** 2).sum()


@torch.no_grad()
def inhibition_efficacy(model, g, gene_idx: int, target_cells: np.ndarray, ref_cells: np.ndarray,
                        niche: np.ndarray, alpha: float = 1.0, base_pred: torch.Tensor | None = None):
    """Return overall efficacy and a dict of per-niche efficacy for one gene."""
    x_f = make_input(g.x, g.readout_idx)
    if base_pred is None:
        base_pred = model(x_f, g)
    x_cf = x_f.clone()
    x_cf[:, gene_idx] = x_cf[:, gene_idx] * (1.0 - alpha)
    pred_cf = model(x_cf, g)
    u, norm2 = state_axis(g.x, g.readout_idx, target_cells, ref_cells)
    delta = (pred_cf - base_pred)[:, g.readout_idx]          # cells x readout
    shift = (delta @ u) / norm2                              # fraction of gap closed per cell
    shift = shift.numpy()
    overall = float(shift[target_cells].mean())
    per_niche = {}
    for n in np.unique(niche[target_cells]):
        cells = target_cells[niche[target_cells] == n]
        if len(cells) >= MIN_NICHE_CELLS:
            per_niche[int(n)] = float(shift[cells].mean())
    return overall, per_niche


def niche_max(per_niche: dict) -> float:
    """Efficacy in the most responsive niche (the niche-specific score used for ranking)."""
    return max(per_niche.values()) if per_niche else 0.0


@torch.no_grad()
def scan_targets(model, g, genes: list[str], target_cells, ref_cells, niche, alpha: float = 1.0,
                 candidates: list[int] | None = None):
    x_f = make_input(g.x, g.readout_idx)
    base = model(x_f, g)
    readout = set(g.readout_idx.tolist())
    idx = candidates if candidates is not None else range(len(genes))
    rows = []
    for j in idx:
        if j in readout:
            rows.append((genes[j], 0.0, {}))       # read-out genes are outputs, not inputs
            continue
        eff, per = inhibition_efficacy(model, g, j, target_cells, ref_cells, niche, alpha, base)
        rows.append((genes[j], eff, per))
    return rows


def selectivity(X_tumour: np.ndarray, type_tumour: np.ndarray, X_normal: np.ndarray,
                type_normal: np.ndarray, n_types: int) -> np.ndarray:
    """Per-gene selectivity in [0, 1).

    For each gene take the lineage c* in which it is most expressed in the
    tumour and compare with the same lineage in matched normal tissue on the
    log scale: sigma = 1 - exp(-max(mu_T - mu_N, 0)). A gene that is not higher
    in the tumour context than in normal tissue gets 0; a log-fold increase of
    1 gives about 0.63.
    """
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


def combine_scores(efficacy: np.ndarray, select: np.ndarray | None, lam: float = 1.0) -> np.ndarray:
    eff = np.clip(efficacy, 0, None)
    eff = eff / (eff.max() + 1e-12)
    if select is None:
        return eff
    return eff * np.power(select, lam)
