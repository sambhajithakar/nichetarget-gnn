"""NumPy implementation of NicheTarget-GNN (same architecture as ``model.py``).

Used where PyTorch is unavailable. The architecture, initialisation, loss,
optimiser (AdamW) and gradient clipping follow ``model.py``; the two
differences are that GELU uses the tanh approximation and that the graph must
have a fixed in-degree (every receiver has exactly k incoming edges, which is
true for the k-nearest-neighbour graphs used here), so the attention softmax
over incoming edges is a softmax over the last axis of an (n, k) array.

The graph is a ``NPGraph`` (see ``np_graph.py``).
"""
from __future__ import annotations

import numpy as np

from .npgrad import (DTYPE, no_grad, AdamW, LayerNorm, Linear, Module, Tensor, concat, dropout,
                     gather_rows, gelu, softmax_last)


class EdgeAttentionLayer(Module):
    def __init__(self, d, d_edge, rng):
        self.q, self.k, self.v = Linear(d, d, rng), Linear(d, d, rng), Linear(d, d, rng)
        self.e = Linear(d_edge, d, rng)
        self.o1, self.o2 = Linear(2 * d, d, rng), Linear(d, d, rng)
        self.norm = LayerNorm(d)
        self.d = d

    def __call__(self, h, g, ef):
        n, kk = h.shape[0], g.k
        k = gather_rows(self.k(h), g.src, g.scatter_src)
        v = gather_rows(self.v(h), g.src, g.scatter_src)
        e = self.e(ef)
        k, v = k + e, v + e
        q = self.q(h)                                   # receivers are dst = repeat(arange(n), k)
        score = (q.reshape(n, 1, self.d) * k.reshape(n, kk, self.d)).sum(-1) * (1.0 / np.sqrt(self.d))
        alpha = softmax_last(score)                     # n x k
        agg = (alpha.reshape(n, kk, 1) * v.reshape(n, kk, self.d)).sum(1)
        out = self.o2(gelu(self.o1(concat([h, agg], -1))))
        return self.norm(h + out), alpha


class SelfMLP(Module):
    def __init__(self, d, rng):
        self.l, self.n = Linear(d, d, rng), LayerNorm(d)

    def __call__(self, h):
        return self.n(gelu(self.l(h)))


class NicheTargetGNN(Module):
    def __init__(self, n_genes, n_types, n_lr, d=128, n_layers=2, use_spatial=True, use_niche=True,
                 use_lr=True, dropout=0.1, seed=0):
        rng = np.random.default_rng(seed)
        self.use_spatial, self.use_niche, self.use_lr = use_spatial, use_niche, use_lr
        self.i1, self.i2 = Linear(n_genes, d, rng), Linear(d, d, rng)
        self.n1, self.n2 = Linear(n_types, d, rng), Linear(d, 2 * d * n_layers, rng)
        d_edge = 1 + (n_lr if use_lr else 0)
        self.layers = [EdgeAttentionLayer(d, d_edge, rng) for _ in range(n_layers)]
        self.self_mlp = [SelfMLP(d, rng) for _ in range(n_layers)]
        self.d1, self.d2 = Linear(d, d, rng), Linear(d, n_genes, rng)
        self.d, self.n_layers, self.p_drop = d, n_layers, dropout
        self.training = False
        self.rng = np.random.default_rng(seed + 1)

    def edge_features(self, x, g):
        feats = [g.edge_dist]
        if self.use_lr and len(g.lr_idx):
            lig = x[g.src][:, g.lr_idx[:, 0]]
            rec = np.repeat(x[:, g.lr_idx[:, 1]], g.k, axis=0)   # receiver of edge e is e // k
            feats.append(lig * rec)
        return Tensor(np.concatenate(feats, -1))

    def __call__(self, x_in: np.ndarray, g, return_attention=False):
        x = Tensor(x_in)
        h = self.i2(dropout(gelu(self.i1(x)), self.p_drop, self.rng, self.training))
        film = None
        if self.use_niche:
            film = self.n2(gelu(self.n1(Tensor(g.niche_comp)))).reshape(-1, self.n_layers, 2, self.d)
        ef = self.edge_features(x_in, g) if self.use_spatial else None
        attn = []
        for l in range(self.n_layers):
            if self.use_spatial:
                h, a = self.layers[l](h, g, ef)
                attn.append(a.data)
            else:
                h = h + self.self_mlp[l](h)
            if film is not None:
                h = (film[:, l, 0] + 1.0) * h + film[:, l, 1]
        out = self.d2(gelu(self.d1(h)))
        return (out, attn) if return_attention else out

    def predict(self, x_in, g):
        self.training = False
        with no_grad():
            return self(x_in, g).data


def make_input(x: np.ndarray, readout_idx, mask=None):
    x_in = x.copy()
    x_in[:, readout_idx] = 0.0
    if mask is not None:
        x_in = x_in * (~mask)
    return x_in


def masked_loss(pred: Tensor, x: np.ndarray, target_mask: np.ndarray, w: np.ndarray):
    """sum(w * mask * (pred - x)^2) / sum(w * mask), as in model.train_model."""
    wm = (target_mask * w).astype(DTYPE)
    denom = float(wm.sum())
    diff = pred - x
    return (diff * diff * (wm / denom)).sum()


def train_model(model: NicheTargetGNN, g, epochs=300, lr=2e-3, mask_frac=0.15, weight_decay=1e-4,
                seed=0, readout_weight=5.0, verbose=False, batches=None):
    """Self-supervised masked-expression training (see model.train_model).

    ``batches``: optional callable returning a list of sub-graphs for one epoch
    (used for large sections); by default the whole graph is one batch.
    """
    rng = np.random.default_rng(seed)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = []
    for ep in range(epochs):
        model.training = True
        ep_loss = []
        for sg in (batches(ep) if batches is not None else [g]):
            x = sg.x
            n, G = x.shape
            is_readout = np.zeros(G, dtype=bool)
            is_readout[sg.readout_idx] = True
            w = np.ones(G, dtype=DTYPE)
            w[is_readout] = readout_weight
            mask = (rng.random((n, G)) < mask_frac) & (~is_readout)
            x_in = make_input(x, sg.readout_idx, mask)
            pred = model(x_in, sg)
            loss = masked_loss(pred, x, mask | is_readout[None, :], w)
            opt.zero_grad()
            loss.backward()
            opt.clip_grad_norm(1.0)
            opt.step()
            ep_loss.append(float(loss.data))
        history.append(float(np.mean(ep_loss)))
        if verbose and (ep % 25 == 0 or ep == epochs - 1):
            print(f"epoch {ep:4d} loss {history[-1]:.4f}", flush=True)
    model.training = False
    return history
