"""Finite-difference gradient check of the NumPy NicheTarget-GNN, and kNN check."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nichetarget import npgrad
npgrad.DTYPE = np.float64
import nichetarget.np_graph as NG
import nichetarget.np_model as NM
NG.DTYPE = NM.DTYPE = np.float64


def test_knn():
    rng = np.random.default_rng(0)
    c = rng.uniform(0, 50, (700, 2))
    d, i = NG.knn(c, 8)
    D = np.sqrt(((c[:, None] - c[None]) ** 2).sum(-1)); np.fill_diagonal(D, np.inf)
    ref = np.sort(D, 1)[:, :8]
    assert np.allclose(d, ref), "knn distances differ"
    print("knn ok")


def test_grad(**kw):
    rng = np.random.default_rng(1)
    n, G, T = 60, 12, 3
    X = rng.gamma(1.0, 1.0, (n, G))
    coords = rng.uniform(0, 10, (n, 2)); ct = rng.integers(0, T, n)
    genes = [f"g{i}" for i in range(G)]
    g = NG.build_graph(X, coords, ct, T, genes, [("g0", "g1"), ("g2", "g3")], ["g10", "g11"], k=4, k_niche=5)
    m = NM.NicheTargetGNN(G, T, 2, d=8, dropout=0.0, seed=0, **kw)
    mask = rng.random((n, G)) < 0.3
    w = np.ones(G); w[[10, 11]] = 5.0
    tm = mask.copy(); tm[:, [10, 11]] = True
    xin = NM.make_input(X, g.readout_idx, mask)
    def f():
        return float(NM.masked_loss(m(xin, g), X, tm, w).data)
    loss = NM.masked_loss(m(xin, g), X, tm, w)
    for p in m.parameters(): p.grad = None
    loss.backward()
    worst = 0
    unused = sum(p.grad is None for p in m.parameters())
    print('params without grad (self_mlp, unused when spatial):', unused)
    for p in m.parameters():
        if p.grad is None: continue
        flat = p.data.reshape(-1)
        for j in rng.choice(flat.size, min(5, flat.size), replace=False):
            old = flat[j]; eps = 1e-6
            flat[j] = old + eps; fp = f(); flat[j] = old - eps; fm = f(); flat[j] = old
            num = (fp - fm) / (2 * eps); ana = p.grad.reshape(-1)[j]
            worst = max(worst, abs(num - ana) / max(1e-4, abs(num) + abs(ana)))
    print("max relative gradient error", worst)
    assert worst < 1e-3


if __name__ == "__main__":
    test_knn(); test_grad(); test_grad(use_spatial=False); test_grad(use_niche=False, use_lr=False)
