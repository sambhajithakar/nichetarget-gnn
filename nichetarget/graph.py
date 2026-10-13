"""Graph construction for NicheTarget-GNN.

A tissue section becomes a directed cell graph (source -> receiver) built from
k nearest spatial neighbours. Each edge carries
  * a distance feature, and
  * one ligand-receptor (LR) feature per curated LR pair, computed on the fly as
    x_src[ligand] * x_dst[receptor]. Because LR features are recomputed from
    the expression matrix in every forward pass, an in-silico inhibition of a
    ligand or receptor automatically removes the corresponding signalling edge.
Each cell also gets a niche descriptor: the cell-type composition of its
spatial neighbourhood.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial import cKDTree


@dataclass
class TissueGraph:
    x: torch.Tensor            # cells x genes (float)
    edge_index: torch.Tensor   # 2 x E (row 0 = source, row 1 = receiver)
    edge_dist: torch.Tensor    # E x 1, distances scaled to [0, 1]
    lr_idx: torch.Tensor       # P x 2 gene indices (ligand, receptor)
    niche_comp: torch.Tensor   # cells x n_types, neighbourhood composition
    cell_type: torch.Tensor    # cells
    readout_idx: torch.Tensor  # genes that define the state read-out (never used as input)


def knn_edges(coords: np.ndarray, k: int = 8):
    tree = cKDTree(coords)
    dist, nbr = tree.query(coords, k=k + 1)
    dist, nbr = dist[:, 1:], nbr[:, 1:]
    n = coords.shape[0]
    dst = np.repeat(np.arange(n), k)
    src = nbr.reshape(-1)
    d = dist.reshape(-1)
    d = d / (d.max() + 1e-8)
    return np.stack([src, dst]), d


def niche_composition(coords: np.ndarray, cell_type: np.ndarray, n_types: int, k: int = 15):
    tree = cKDTree(coords)
    _, nbr = tree.query(coords, k=k + 1)        # includes the cell itself
    onehot = np.eye(n_types)[cell_type]
    return onehot[nbr].mean(1)


def build_graph(X: np.ndarray, coords: np.ndarray, cell_type: np.ndarray, n_types: int,
                genes: list[str], lr_pairs: list[tuple[str, str]], readout_genes: list[str],
                k: int = 8, k_niche: int = 15) -> TissueGraph:
    gi = {g: i for i, g in enumerate(genes)}
    ei, d = knn_edges(coords, k)
    lr = [(gi[l], gi[r]) for l, r in lr_pairs if l in gi and r in gi]
    comp = niche_composition(coords, cell_type, n_types, k_niche)
    return TissueGraph(
        x=torch.tensor(X, dtype=torch.float32),
        edge_index=torch.tensor(ei, dtype=torch.long),
        edge_dist=torch.tensor(d, dtype=torch.float32).unsqueeze(1),
        lr_idx=torch.tensor(lr, dtype=torch.long).reshape(-1, 2),
        niche_comp=torch.tensor(comp, dtype=torch.float32),
        cell_type=torch.tensor(cell_type, dtype=torch.long),
        readout_idx=torch.tensor([gi[g] for g in readout_genes], dtype=torch.long),
    )
