"""Simulated tumour tissues with planted, niche-specific causal drivers.

The simulator is used only to test whether a method can recover known causal
targets. Ground truth is defined by the generative process below.

Generative process (one tissue section)
---------------------------------------
* Cells are placed in 2D. Three spatial niches are laid out as disks:
  - "myeloid" niche: malignant cells mixed with M2-like macrophages
  - "stromal" niche: malignant cells mixed with cancer-associated fibroblasts
  - "immune" niche: malignant cells mixed with T cells
  plus a background of malignant + endothelial cells.
* Malignant cells have a latent aggressiveness a_i in (0, 1) driven by
    s_i = w1 * sum_{j in N(i)} L1_j * R1_i      (macrophage -> tumour paracrine)
        + w2 * sum_{j in N(i)} L2_j * R2_i      (fibroblast -> tumour paracrine)
        + w3 * D1_i                            (cell-intrinsic driver)
        + w4 * E1_i                            (driver that is also high in normal tissue)
        + noise
    a_i = sigmoid(s_i - threshold)
* Aggressive-programme genes (the state read-out) rise with a_i.
* "Passenger" genes are co-regulated with a_i but have no causal effect.
* Counts are drawn from a negative binomial and log-normalised.

The true targets that shift malignant cells from the aggressive to the
non-aggressive state are L1, R1, L2, R2, D1 and E1. E1 is a non-selective
target because it is highly expressed in a matched normal-tissue reference.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CELL_TYPES = ["malignant", "macrophage", "fibroblast", "tcell", "endothelial"]


@dataclass
class SimTissue:
    X: np.ndarray                 # cells x genes, log1p-normalised
    counts: np.ndarray            # cells x genes, raw counts
    coords: np.ndarray            # cells x 2
    cell_type: np.ndarray         # int index into CELL_TYPES
    niche: np.ndarray             # int: 0 background, 1 myeloid, 2 stromal, 3 immune
    aggressiveness: np.ndarray    # latent a_i (nan for non-malignant cells)
    genes: list[str]
    lr_pairs: list[tuple[str, str]]
    true_targets: list[str]
    selective_targets: list[str]
    readout_genes: list[str]
    passenger_genes: list[str]
    niche_specific: dict[str, int] = field(default_factory=dict)  # gene -> niche id
    X_normal: np.ndarray | None = None        # normal reference, cells x genes
    normal_cell_type: np.ndarray | None = None


def _nb(rng, mu, theta=5.0):
    p = theta / (theta + mu)
    return rng.negative_binomial(theta, p)


def simulate_tissue(n_cells: int = 3000, n_genes: int = 200, seed: int = 0,
                    effect: float = 1.0, k_neighbors: int = 8,
                    n_decoy_lr: int = 6) -> SimTissue:
    rng = np.random.default_rng(seed)

    # ---- genes ---------------------------------------------------------
    named = ["L1", "R1", "L2", "R2", "D1", "E1"]
    readout = [f"AGG{i}" for i in range(15)]
    passengers = [f"PASS{i}" for i in range(10)]
    markers = {ct: [f"{ct[:3].upper()}M{i}" for i in range(8)] for ct in CELL_TYPES}
    decoy_lr = [(f"DL{i}", f"DR{i}") for i in range(n_decoy_lr)]
    decoy_flat = [g for p in decoy_lr for g in p]
    used = named + readout + passengers + [g for v in markers.values() for g in v] + decoy_flat
    filler = [f"G{i}" for i in range(max(0, n_genes - len(used)))]
    genes = used + filler
    gi = {g: i for i, g in enumerate(genes)}
    G = len(genes)

    # ---- spatial layout ----------------------------------------------------
    coords = rng.uniform(0, 100, size=(n_cells, 2))
    centres = {1: (25, 25), 2: (75, 30), 3: (50, 78)}
    radius = 17.0
    niche = np.zeros(n_cells, dtype=int)
    for nid, c in centres.items():
        d = np.linalg.norm(coords - np.array(c), axis=1)
        niche[d < radius] = nid

    cell_type = np.zeros(n_cells, dtype=int)
    type_probs = {
        0: [0.80, 0.03, 0.03, 0.04, 0.10],
        1: [0.55, 0.40, 0.02, 0.02, 0.01],
        2: [0.55, 0.02, 0.40, 0.02, 0.01],
        3: [0.55, 0.02, 0.02, 0.40, 0.01],
    }
    for nid, p in type_probs.items():
        idx = np.where(niche == nid)[0]
        cell_type[idx] = rng.choice(len(CELL_TYPES), size=len(idx), p=p)

    # ---- baseline means ------------------------------------------------------
    base = rng.gamma(2.0, 0.5, size=G)            # low background for every gene
    mu = np.tile(base, (n_cells, 1))
    for t, ct in enumerate(CELL_TYPES):
        rows = cell_type == t
        for g in markers[ct]:
            mu[rows, gi[g]] += rng.uniform(6, 12)

    mal = cell_type == 0
    mac = cell_type == 1
    fib = cell_type == 2
    tc = cell_type == 3

    # ligands / receptors / drivers
    mu[mac, gi["L1"]] += 10.0
    mu[fib, gi["L2"]] += 10.0
    mu[mal, gi["R1"]] += rng.gamma(4, 1.5, size=mal.sum())
    mu[mal, gi["R2"]] += rng.gamma(4, 1.5, size=mal.sum())
    mu[mal, gi["D1"]] += rng.gamma(2, 2.5, size=mal.sum())
    mu[mal, gi["E1"]] += rng.gamma(2, 2.0, size=mal.sum())
    # decoy LR pairs: expressed by stromal/immune and tumour, but no effect
    for (dl, dr) in decoy_lr:
        sender = rng.choice([1, 2, 3])
        mu[cell_type == sender, gi[dl]] += 8.0
        mu[mal, gi[dr]] += rng.gamma(4, 1.5, size=mal.sum())

    # draw upstream counts first so the latent state depends on realised values
    counts = _nb(rng, mu).astype(float)

    # ---- neighbour graph for the generative signal ---------------------------
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(coords)
        _, nbr = tree.query(coords, k=k_neighbors + 1)
        nbr = nbr[:, 1:]
    except ImportError:                       # exact kNN without SciPy (same neighbours)
        from .np_graph import knn
        nbr = knn(coords, k_neighbors)[1]

    def norm(v):
        return np.log1p(v)

    L1 = norm(counts[:, gi["L1"]]); R1 = norm(counts[:, gi["R1"]])
    L2 = norm(counts[:, gi["L2"]]); R2 = norm(counts[:, gi["R2"]])
    D1 = norm(counts[:, gi["D1"]]); E1 = norm(counts[:, gi["E1"]])
    sig1 = L1[nbr].sum(1) * R1
    sig2 = L2[nbr].sum(1) * R2
    s = (effect * (0.12 * sig1 + 0.12 * sig2 + 0.9 * D1 + 0.7 * E1)
         + rng.normal(0, 0.6, size=n_cells))
    thr = np.quantile(s[mal], 0.55)
    a = 1.0 / (1.0 + np.exp(-(s - thr) * 1.5))
    a[~mal] = np.nan

    # ---- downstream programme (read-out) and passengers ----------------------
    mu2 = mu.copy()
    am = np.nan_to_num(a)
    for g in readout:
        mu2[mal, gi[g]] += 12.0 * am[mal]
    for g in passengers:
        mu2[mal, gi[g]] += 6.0 * am[mal] + rng.gamma(1, 1, size=mal.sum())
    down_idx = [gi[g] for g in readout + passengers]
    counts[:, down_idx] = _nb(rng, mu2[:, down_idx])

    lib = counts.sum(1, keepdims=True)
    X = np.log1p(counts / lib * 1e3)

    # ---- matched normal reference (same lineages, no tumour signalling) ------
    n_norm = 800
    norm_type = rng.choice([0, 1, 2, 3, 4], size=n_norm, p=[0.5, 0.15, 0.15, 0.1, 0.1])
    mu_n = np.tile(base, (n_norm, 1))
    for t, ct in enumerate(CELL_TYPES):
        rows = norm_type == t
        for g in markers[ct]:
            mu_n[rows, gi[g]] += rng.uniform(6, 12)
    epi = norm_type == 0  # normal epithelium, lineage-matched to malignant cells
    mu_n[epi, gi["E1"]] += 14.0      # E1 is high in normal epithelium -> not selective
    mu_n[epi, gi["R1"]] += 1.0
    mu_n[epi, gi["R2"]] += 1.0
    mu_n[epi, gi["D1"]] += 0.5
    mu_n[norm_type == 1, gi["L1"]] += 2.0
    mu_n[norm_type == 2, gi["L2"]] += 2.0
    cn = _nb(rng, mu_n).astype(float)
    Xn = np.log1p(cn / cn.sum(1, keepdims=True) * 1e3)

    return SimTissue(
        X=X.astype(np.float32), counts=counts, coords=coords.astype(np.float32),
        cell_type=cell_type, niche=niche, aggressiveness=a, genes=genes,
        lr_pairs=[("L1", "R1"), ("L2", "R2")] + decoy_lr,
        true_targets=["L1", "R1", "L2", "R2", "D1", "E1"],
        selective_targets=["L1", "R1", "L2", "R2", "D1"],
        readout_genes=readout, passenger_genes=passengers,
        niche_specific={"L1": 1, "R1": 1, "L2": 2, "R2": 2},
        X_normal=Xn.astype(np.float32), normal_cell_type=norm_type,
    )
