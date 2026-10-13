"""Graph construction and small numerical utilities without SciPy/scikit-learn.

* exact k-nearest neighbours in 2D by grid bucketing,
* the NicheTarget-GNN tissue graph (fixed in-degree k) and spatial tiling,
* PCA, k-means, average ranks, Spearman correlation and average precision.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .npgrad import DTYPE


# --------------------------------------------------------------------- kNN
def knn(coords: np.ndarray, k: int, include_self: bool = False):
    """Exact kNN in 2D. Returns (dist, idx), each n x k, sorted by distance."""
    coords = np.asarray(coords, dtype=np.float64)
    n = len(coords)
    kk = k + 1
    lo = coords.min(0)
    area = np.prod(coords.max(0) - lo) + 1e-9
    w = np.sqrt(area / n * max(kk, 4))           # about kk points per bin
    b = np.floor((coords - lo) / w).astype(np.int64)
    nbx, nby = b[:, 0].max() + 1, b[:, 1].max() + 1
    key = b[:, 0] * nby + b[:, 1]
    order = np.argsort(key, kind="stable")
    starts = np.searchsorted(key[order], np.arange(nbx * nby))
    ends = np.searchsorted(key[order], np.arange(nbx * nby), side="right")
    out_d = np.full((n, kk), np.inf)
    out_i = np.zeros((n, kk), dtype=np.int64)
    todo = np.ones(n, dtype=bool)
    r = 1
    while todo.any():
        for bk in np.unique(key[todo]):
            bx, by = divmod(int(bk), nby)
            members = order[starts[bk]:ends[bk]]
            members = members[todo[members]]
            if len(members) == 0:
                continue
            cand = [order[starts[x * nby + y]:ends[x * nby + y]]
                    for x in range(max(bx - r, 0), min(bx + r, nbx - 1) + 1)
                    for y in range(max(by - r, 0), min(by + r, nby - 1) + 1)]
            cand = np.concatenate(cand)
            d = np.sqrt(((coords[members, None, :] - coords[None, cand, :]) ** 2).sum(-1))
            m = min(kk, len(cand))
            part = np.argpartition(d, m - 1, axis=1)[:, :m] if len(cand) > m else np.tile(np.arange(len(cand)), (len(members), 1))
            dd = np.take_along_axis(d, part, 1)
            s = np.argsort(dd, 1)
            out_d[members, :m] = np.take_along_axis(dd, s, 1)
            out_i[members, :m] = cand[np.take_along_axis(part, s, 1)]
            # exact if the kth distance is within the guaranteed search radius r*w
            ok = (m == kk) & (out_d[members, kk - 1] <= r * w)
            todo[members[ok]] = False
        r += 1
        if r > max(nbx, nby) + 1:
            break
    if include_self:
        return out_d[:, :k], out_i[:, :k]
    # drop self (distance 0, normally first; handle duplicates robustly)
    is_self = out_i == np.arange(n)[:, None]
    has_self = is_self.any(1)
    # rows where self appears: remove it; else drop the last column
    keep = ~is_self
    keep[~has_self, kk - 1] = False
    d_res = out_d[keep].reshape(n, k)
    i_res = out_i[keep].reshape(n, k)
    return d_res, i_res


# --------------------------------------------------------------------- graph
@dataclass
class NPGraph:
    x: np.ndarray            # cells x genes, float32
    src: np.ndarray          # E, source of each edge; receiver of edge e is e // k
    k: int
    edge_dist: np.ndarray    # E x 1
    lr_idx: np.ndarray       # P x 2
    niche_comp: np.ndarray   # cells x n_types
    readout_idx: np.ndarray
    cell_type: np.ndarray | None = None

    def __post_init__(self):
        n = self.x.shape[0]
        self._order = np.argsort(self.src, kind="stable")
        counts = np.bincount(self.src, minlength=n)
        self._has = np.where(counts > 0)[0]
        self._starts = np.concatenate([[0], np.cumsum(counts)[:-1]])[self._has]

    def scatter_src(self, g):
        """Sum edge-level gradients (E x d) onto source nodes (n x d)."""
        out = np.zeros((self.x.shape[0],) + g.shape[1:], dtype=DTYPE)
        if len(self._has):
            out[self._has] = np.add.reduceat(g[self._order], self._starts, axis=0)
        return out

    @property
    def n(self):
        return self.x.shape[0]


def niche_composition(nbr_incl_self: np.ndarray, cell_type: np.ndarray, n_types: int):
    onehot = np.eye(n_types, dtype=DTYPE)[cell_type]
    return onehot[nbr_incl_self].mean(1)


def build_graph(X, coords, cell_type, n_types, genes, lr_pairs, readout_genes, k=8, k_niche=15,
                dist_scale=None):
    gi = {g: i for i, g in enumerate(genes)}
    d, nbr = knn(coords, k)
    n = len(coords)
    src = nbr.reshape(-1)
    dd = d.reshape(-1)
    scale = dist_scale if dist_scale is not None else dd.max() + 1e-8
    dd = np.clip(dd / scale, 0, None)
    lr = np.array([(gi[l], gi[r]) for l, r in lr_pairs if l in gi and r in gi], dtype=np.int64).reshape(-1, 2)
    _, nb2 = knn(coords, k_niche + 1, include_self=True)
    comp = niche_composition(nb2, cell_type, n_types)
    return NPGraph(x=np.asarray(X, dtype=DTYPE), src=src, k=k, edge_dist=dd[:, None].astype(DTYPE), lr_idx=lr,
                   niche_comp=comp, readout_idx=np.array([gi[g] for g in readout_genes], dtype=np.int64),
                   cell_type=np.asarray(cell_type))


def subgraph(g: NPGraph, nodes: np.ndarray, x: np.ndarray | None = None) -> NPGraph:
    """Induced sub-graph on ``nodes`` keeping fixed in-degree.

    Edges from sources outside ``nodes`` are redirected to the receiver itself
    (self-loop with its original distance) so each receiver keeps k incoming
    edges; callers use halos so that the cells they read out are interior.
    """
    nodes = np.asarray(nodes)
    pos = -np.ones(g.n, dtype=np.int64)
    pos[nodes] = np.arange(len(nodes))
    e = (nodes[:, None] * g.k + np.arange(g.k)[None, :]).reshape(-1)
    src = pos[g.src[e]]
    recv = np.repeat(np.arange(len(nodes)), g.k)
    src = np.where(src < 0, recv, src)
    return NPGraph(x=(g.x if x is None else x)[nodes], src=src, k=g.k, edge_dist=g.edge_dist[e],
                   lr_idx=g.lr_idx, niche_comp=g.niche_comp[nodes], readout_idx=g.readout_idx,
                   cell_type=None if g.cell_type is None else g.cell_type[nodes])


def hops(g: NPGraph, seeds: np.ndarray, n_hops: int) -> np.ndarray:
    """Seeds plus all cells within n_hops incoming-edge steps (message sources)."""
    cur = np.zeros(g.n, dtype=bool)
    cur[seeds] = True
    for _ in range(n_hops):
        recv = np.where(cur)[0]
        e = (recv[:, None] * g.k + np.arange(g.k)[None, :]).reshape(-1)
        cur[g.src[e]] = True
    return np.where(cur)[0]


def spatial_tiles(coords, tile_size):
    b = np.floor((coords - coords.min(0)) / tile_size).astype(np.int64)
    key = b[:, 0] * (b[:, 1].max() + 1) + b[:, 1]
    return [np.where(key == u)[0] for u in np.unique(key)]


# --------------------------------------------------------------------- stats
def pca(X, n_components, seed=0):
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc.astype(np.float64), full_matrices=False)
    return (U[:, :n_components] * S[:n_components]), Vt[:n_components]


def kmeans(Z, k, n_init=10, seed=0, iters=300):
    """Lloyd's algorithm with k-means++ initialisation; best of n_init runs."""
    rng = np.random.default_rng(seed)
    Z = np.asarray(Z, dtype=np.float64)
    best, best_inertia = None, np.inf
    for _ in range(n_init):
        c = [Z[rng.integers(len(Z))]]
        for _ in range(1, k):
            d2 = np.min([((Z - ci) ** 2).sum(1) for ci in c], 0)
            c.append(Z[rng.choice(len(Z), p=d2 / d2.sum())])
        C = np.array(c)
        lab = None
        for _ in range(iters):
            d = ((Z[:, None, :] - C[None]) ** 2).sum(-1) if len(Z) * k < 5e7 else np.stack([((Z - ci) ** 2).sum(1) for ci in C], 1)
            new = d.argmin(1)
            if lab is not None and (new == lab).all():
                break
            lab = new
            for j in range(k):
                if (lab == j).any():
                    C[j] = Z[lab == j].mean(0)
        inertia = d[np.arange(len(Z)), lab].sum()
        if inertia < best_inertia:
            best, best_inertia = lab.copy(), inertia
    return best


def rankdata(a):
    return pd.Series(np.asarray(a, dtype=np.float64)).rank(method="average").to_numpy()


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def spearman_columns(M, y):
    """Spearman correlation of every column of M with y."""
    ry = rankdata(y)
    ry = (ry - ry.mean()) / (ry.std() + 1e-12)
    R = pd.DataFrame(M).rank(method="average").to_numpy()
    sd = R.std(0)
    R = (R - R.mean(0)) / np.where(sd > 0, sd, 1)
    out = (R * ry[:, None]).mean(0)
    out[sd == 0] = 0.0
    return out


def average_precision(y, s):
    """Same definition as sklearn.metrics.average_precision_score (ties grouped)."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=np.float64)
    order = np.argsort(-s, kind="mergesort")
    y, s = y[order], s[order]
    distinct = np.where(np.diff(s))[0]
    thr = np.r_[distinct, len(y) - 1]
    tps = np.cumsum(y)[thr]
    fps = 1 + thr - tps
    prec = tps / (tps + fps)
    rec = tps / y.sum()
    rec = np.r_[0, rec]
    return float(np.sum(np.diff(rec) * prec))


def adjusted_rand(a, b):
    a, b = np.asarray(a), np.asarray(b)
    ct = pd.crosstab(a, b).to_numpy()
    comb = lambda x: x * (x - 1) / 2  # noqa: E731
    sij = comb(ct).sum()
    sa, sb = comb(ct.sum(1)).sum(), comb(ct.sum(0)).sum()
    exp = sa * sb / comb(len(a))
    mx = 0.5 * (sa + sb)
    return float((sij - exp) / (mx - exp)) if mx != exp else 1.0
