"""Minimal reverse-mode automatic differentiation in NumPy.

This module exists so that NicheTarget-GNN can be trained in environments
where PyTorch is unavailable. It implements only the operations the model
needs (dense layers, GELU, LayerNorm, row gathers, softmax over a fixed number
of incoming edges, reshapes, slicing and elementwise arithmetic). Every
operation records a backward closure; ``Tensor.backward`` runs them in reverse
topological order. Gradients are checked against finite differences in
``tests/test_npgrad.py``.
"""
from __future__ import annotations

import numpy as np

DTYPE = np.float32
GRAD_ENABLED = True   # set False for inference: no graph is recorded, intermediates are freed at once


class no_grad:
    def __enter__(self):
        global GRAD_ENABLED
        self.prev, GRAD_ENABLED = GRAD_ENABLED, False

    def __exit__(self, *exc):
        global GRAD_ENABLED
        GRAD_ENABLED = self.prev


def _unbroadcast(g, shape):
    while g.ndim > len(shape):
        g = g.sum(0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(i, keepdims=True)
    return g


class Tensor:
    __slots__ = ("data", "grad", "_parents", "_backward", "requires_grad")

    def __init__(self, data, parents=(), backward=None, requires_grad=False):
        self.data = np.asarray(data, dtype=DTYPE)
        self.grad = None
        if not GRAD_ENABLED:
            parents = ()
        self._parents = parents
        self._backward = backward
        self.requires_grad = requires_grad or any(p.requires_grad for p in parents)

    # ------------------------------------------------------------------ utils
    @property
    def shape(self):
        return self.data.shape

    def _acc(self, g):
        if not self.requires_grad:
            return
        self.grad = g if self.grad is None else self.grad + g

    def backward(self, grad=None):
        topo, seen = [], set()

        def visit(t):
            stack = [(t, False)]
            while stack:
                node, done = stack.pop()
                if done:
                    topo.append(node)
                    continue
                if id(node) in seen:
                    continue
                seen.add(id(node))
                stack.append((node, True))
                for p in node._parents:
                    if p.requires_grad and id(p) not in seen:
                        stack.append((p, False))

        visit(self)
        self.grad = np.ones_like(self.data) if grad is None else grad
        for node in reversed(topo):
            if node._backward is not None and node.grad is not None:
                node._backward(node.grad)
                if node._parents:          # free intermediate gradients
                    node.grad = None if node is not self else node.grad

    # ------------------------------------------------------------- arithmetic
    def __add__(self, o):
        o = o if isinstance(o, Tensor) else Tensor(o)
        out = Tensor(self.data + o.data, (self, o))

        def bw(g):
            self._acc(_unbroadcast(g, self.shape))
            o._acc(_unbroadcast(g, o.shape))
        out._backward = bw
        return out

    __radd__ = __add__

    def __mul__(self, o):
        o = o if isinstance(o, Tensor) else Tensor(o)
        out = Tensor(self.data * o.data, (self, o))

        def bw(g):
            if self.requires_grad:
                self._acc(_unbroadcast(g * o.data, self.shape))
            if o.requires_grad:
                o._acc(_unbroadcast(g * self.data, o.shape))
        out._backward = bw
        return out

    __rmul__ = __mul__

    def __neg__(self):
        return self * -1.0

    def __sub__(self, o):
        return self + (-(o if isinstance(o, Tensor) else Tensor(o)))

    def __matmul__(self, o):
        out = Tensor(self.data @ o.data, (self, o))

        def bw(g):
            if self.requires_grad:
                self._acc(g @ o.data.T)
            if o.requires_grad:
                o._acc(self.data.T @ g)
        out._backward = bw
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,))

        def bw(g):
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self._acc(np.broadcast_to(g, self.shape).astype(DTYPE))
        out._backward = bw
        return out

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), (self,))
        out._backward = lambda g: self._acc(g.reshape(self.shape))
        return out

    def __getitem__(self, key):
        out = Tensor(self.data[key], (self,))

        def bw(g):
            z = np.zeros_like(self.data)
            z[key] = g
            self._acc(z)
        out._backward = bw
        return out


def parameter(data):
    return Tensor(data, requires_grad=True)


# ---------------------------------------------------------------- functions
def concat(ts, axis=-1):
    out = Tensor(np.concatenate([t.data for t in ts], axis=axis), tuple(ts))
    sizes = np.cumsum([t.shape[axis] for t in ts])[:-1]

    def bw(g):
        for t, gg in zip(ts, np.split(g, sizes, axis=axis)):
            t._acc(gg)
    out._backward = bw
    return out


_C = np.float32(np.sqrt(2.0 / np.pi))


def gelu(x):
    """GELU, tanh approximation (Hendrycks & Gimpel)."""
    a = x.data
    inner = _C * (a + 0.044715 * a ** 3)
    t = np.tanh(inner)
    out = Tensor(0.5 * a * (1.0 + t), (x,))

    def bw(g):
        d = 0.5 * (1.0 + t) + 0.5 * a * (1.0 - t ** 2) * _C * (1.0 + 3 * 0.044715 * a ** 2)
        x._acc(g * d)
    out._backward = bw
    return out


def layer_norm(x, w, b, eps=1e-5):
    a = x.data
    mu = a.mean(-1, keepdims=True)
    xc = a - mu
    var = (xc ** 2).mean(-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    xh = xc * inv
    out = Tensor(xh * w.data + b.data, (x, w, b))

    def bw(g):
        if w.requires_grad:
            w._acc((g * xh).reshape(-1, a.shape[-1]).sum(0))
            b._acc(g.reshape(-1, a.shape[-1]).sum(0))
        gx = g * w.data
        n = a.shape[-1]
        x._acc(inv / n * (n * gx - gx.sum(-1, keepdims=True) - xh * (gx * xh).sum(-1, keepdims=True)))
    out._backward = bw
    return out


def gather_rows(x, idx, scatter):
    """x[idx] where ``scatter(g)`` sums edge gradients back onto rows."""
    out = Tensor(x.data[idx], (x,))
    out._backward = lambda g: x._acc(scatter(g))
    return out


def softmax_last(x):
    a = x.data
    e = np.exp(a - a.max(-1, keepdims=True))
    s = e / e.sum(-1, keepdims=True)
    out = Tensor(s, (x,))
    out._backward = lambda g: x._acc(s * (g - (g * s).sum(-1, keepdims=True)))
    return out


def dropout(x, p, rng, training):
    if not training or p <= 0:
        return x
    m = (rng.random(x.shape) >= p).astype(DTYPE) / (1.0 - p)
    return x * m


# ---------------------------------------------------------------- layers
class Module:
    def parameters(self):
        ps = []
        for v in self.__dict__.values():
            if isinstance(v, Tensor) and v.requires_grad:
                ps.append(v)
            elif isinstance(v, Module):
                ps.extend(v.parameters())
            elif isinstance(v, (list, tuple)):
                for u in v:
                    if isinstance(u, Module):
                        ps.extend(u.parameters())
        return ps

    def state(self):
        return [p.data.copy() for p in self.parameters()]

    def load(self, state):
        for p, s in zip(self.parameters(), state):
            p.data[...] = s


class Linear(Module):
    """Same initialisation as torch.nn.Linear (uniform +-1/sqrt(fan_in))."""

    def __init__(self, n_in, n_out, rng):
        bound = 1.0 / np.sqrt(n_in)
        self.W = parameter(rng.uniform(-bound, bound, size=(n_in, n_out)))
        self.b = parameter(rng.uniform(-bound, bound, size=(n_out,)))

    def __call__(self, x):
        return x @ self.W + self.b


class LayerNorm(Module):
    def __init__(self, d):
        self.w = parameter(np.ones(d))
        self.b = parameter(np.zeros(d))

    def __call__(self, x):
        return layer_norm(x, self.w, self.b)


class AdamW:
    """torch.optim.AdamW defaults (betas 0.9/0.999, eps 1e-8, decoupled decay)."""

    def __init__(self, params, lr=2e-3, weight_decay=1e-4, betas=(0.9, 0.999), eps=1e-8):
        self.p, self.lr, self.wd, self.b1, self.b2, self.eps = params, lr, weight_decay, *betas, eps
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        self.t = 0

    def zero_grad(self):
        for p in self.p:
            p.grad = None

    def clip_grad_norm(self, max_norm):
        tot = np.sqrt(sum(float((p.grad.astype(np.float64) ** 2).sum()) for p in self.p if p.grad is not None))
        if tot > max_norm:
            c = max_norm / (tot + 1e-6)
            for p in self.p:
                if p.grad is not None:
                    p.grad = p.grad * c
        return tot

    def step(self):
        self.t += 1
        bc1, bc2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for p, m, v in zip(self.p, self.m, self.v):
            if p.grad is None:
                continue
            g = p.grad
            p.data *= (1 - self.lr * self.wd)
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            p.data -= self.lr * (m / bc1) / (np.sqrt(v / bc2) + self.eps)
