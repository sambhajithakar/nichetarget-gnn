"""NicheTarget-GNN encoder-decoder.

Architecture
------------
1. Input projection of each cell's (masked) expression profile.
2. Niche encoder: MLP over neighbourhood cell-type composition, producing
   FiLM parameters (gamma, beta) that modulate every layer, so that the same
   cell state can respond differently in different niches.
3. L edge-aware graph attention layers over the spatial cell graph. Messages
   use the source embedding plus edge features (distance and ligand-receptor
   co-expression), so paracrine signalling enters the receiver explicitly.
4. Linear decoder back to gene expression.

Setting ``use_spatial=False`` removes message passing (cell-intrinsic model);
``use_niche=False`` removes FiLM conditioning; ``use_lr=False`` drops LR edge
features. These switches implement the ablations reported in the paper.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def scatter_softmax(scores: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    smax = torch.full((n,), -1e9, device=scores.device).scatter_reduce(
        0, index, scores, reduce="amax", include_self=True)
    ex = torch.exp(scores - smax[index])
    denom = torch.zeros(n, device=scores.device).index_add_(0, index, ex) + 1e-16
    return ex / denom[index]


class EdgeAttentionLayer(nn.Module):
    def __init__(self, d: int, d_edge: int):
        super().__init__()
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.e = nn.Linear(d_edge, d) if d_edge > 0 else None
        self.out = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d)
        self.d = d

    def forward(self, h, edge_index, edge_feat):
        src, dst = edge_index
        k = self.k(h)[src]
        v = self.v(h)[src]
        if self.e is not None:
            ef = self.e(edge_feat)
            k = k + ef
            v = v + ef
        score = (self.q(h)[dst] * k).sum(-1) / self.d ** 0.5
        alpha = scatter_softmax(score, dst, h.size(0))
        agg = torch.zeros_like(h).index_add_(0, dst, alpha.unsqueeze(-1) * v)
        return self.norm(h + self.out(torch.cat([h, agg], -1))), alpha


class NicheTargetGNN(nn.Module):
    def __init__(self, n_genes: int, n_types: int, n_lr: int, d: int = 128, n_layers: int = 2,
                 use_spatial: bool = True, use_niche: bool = True, use_lr: bool = True,
                 dropout: float = 0.1):
        super().__init__()
        self.use_spatial, self.use_niche, self.use_lr = use_spatial, use_niche, use_lr
        self.inp = nn.Sequential(nn.Linear(n_genes, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, d))
        self.niche = nn.Sequential(nn.Linear(n_types, d), nn.GELU(), nn.Linear(d, 2 * d * n_layers))
        d_edge = 1 + (n_lr if use_lr else 0)
        self.layers = nn.ModuleList([EdgeAttentionLayer(d, d_edge) for _ in range(n_layers)])
        self.self_mlp = nn.ModuleList([nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.LayerNorm(d))
                                       for _ in range(n_layers)])
        self.dec = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, n_genes))
        self.d, self.n_layers = d, n_layers

    def edge_features(self, x, g):
        feats = [g.edge_dist]
        if self.use_lr and g.lr_idx.numel() > 0:
            src, dst = g.edge_index
            lig = x[src][:, g.lr_idx[:, 0]]
            rec = x[dst][:, g.lr_idx[:, 1]]
            feats.append(lig * rec)
        return torch.cat(feats, -1)

    def forward(self, x_in, g, return_attention: bool = False):
        h = self.inp(x_in)
        film = self.niche(g.niche_comp).view(-1, self.n_layers, 2, self.d) if self.use_niche else None
        ef = self.edge_features(x_in, g)
        attn = []
        for l in range(self.n_layers):
            if self.use_spatial:
                h, a = self.layers[l](h, g.edge_index, ef)
                attn.append(a)
            else:
                h = h + self.self_mlp[l](h)
            if film is not None:
                h = (1 + film[:, l, 0]) * h + film[:, l, 1]
        out = self.dec(h)
        return (out, attn) if return_attention else out


def make_input(x: torch.Tensor, readout_idx: torch.Tensor, mask: torch.Tensor | None = None):
    """Zero the read-out genes (always) and an optional random mask."""
    x_in = x.clone()
    x_in[:, readout_idx] = 0.0
    if mask is not None:
        x_in = x_in * (~mask)
    return x_in


def train_model(model: NicheTargetGNN, g, epochs: int = 300, lr: float = 2e-3,
                mask_frac: float = 0.15, weight_decay: float = 1e-4, seed: int = 0,
                readout_weight: float = 5.0, verbose: bool = False):
    """Self-supervised masked-expression training.

    Read-out genes are hidden from the input for every cell and must be
    predicted from the remaining genes of the cell and its neighbours. A random
    15% of the other genes is additionally masked per step.
    """
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    x = g.x
    n, G = x.shape
    is_readout = torch.zeros(G, dtype=torch.bool)
    is_readout[g.readout_idx] = True
    w = torch.ones(G)
    w[is_readout] = readout_weight
    history = []
    for ep in range(epochs):
        model.train()
        mask = (torch.rand(n, G) < mask_frac) & (~is_readout)
        x_in = make_input(x, g.readout_idx, mask)
        pred = model(x_in, g)
        target_mask = mask | is_readout.unsqueeze(0)
        err = ((pred - x) ** 2) * w
        loss = (err * target_mask).sum() / (target_mask * w).sum()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        history.append(loss.item())
        if verbose and ep % 50 == 0:
            print(f"epoch {ep:4d} loss {loss.item():.4f}")
    model.eval()
    return history
