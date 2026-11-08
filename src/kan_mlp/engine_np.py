"""NumPy re-implementation of the repository's PyTorch MLP and KAN models.

Why it exists: the v2 study was run in an environment without PyTorch. The
engine reproduces models.py exactly in structure (same tensors, same parameter
count, same initialisation distributions, same basis functions) and the
training loop of study.py (full-batch Adam with L2 weight decay, gradient-norm
clipping at 10, ReduceLROnPlateau on validation loss, plateau stopping and
best-checkpoint restoration). Gradients are hand-derived and checked against
finite differences in tests/test_engine_np.py; when PyTorch is installed (CI),
the same file also checks forward values and gradients against models.py.

Random streams differ from PyTorch, so individual runs are not bit-identical to
torch runs; the distributions they are drawn from are identical.
"""
import math
import time
import numpy as np

from .arch import parameter_count


def silu(x):
    s = 1.0 / (1.0 + np.exp(-x))
    return x * s, s * (1.0 + x * (1.0 - s))  # value, derivative


def _act(kind, z):
    if kind == "tanh":
        t = np.tanh(z); return t, 1.0 - t * t
    if kind == "relu":
        return np.maximum(z, 0), (z > 0).astype(z.dtype)
    if kind == "silu":
        return silu(z)
    raise ValueError(kind)


class Basis:
    def __init__(self, family, terms, dtype):
        if terms < 4 or terms % 2:
            raise ValueError("Use an even number of basis terms >=4")
        self.family, self.terms = family, terms
        self.centers = np.linspace(-1, 1, terms).astype(dtype)
        self.h = 2.0 / (terms - 1)
        self.freq = (math.pi * np.arange(1, terms // 2 + 1)).astype(dtype)

    def __call__(self, x):
        """x: (b, i) -> values, derivatives, both (b, i, R)."""
        if self.family == "fourier":
            ph = x[..., None] * self.freq
            s, c = np.sin(ph), np.cos(ph)
            return np.concatenate([s, c], -1), np.concatenate([c * self.freq, -s * self.freq], -1)
        z = (x[..., None] - self.centers) / self.h
        if self.family == "rbf":
            v = np.exp(-0.5 * z * z)
            return v, v * (-z) / self.h
        if self.family != "spline":
            raise ValueError(self.family)
        # Cardinal cubic B-spline: B(a) = [(2-a)_+^3 - 4(1-a)_+^3] / 6, a = |z|.
        a = np.abs(z)
        t2 = np.maximum(2 - a, 0); t1 = np.maximum(1 - a, 0)
        t2s, t1s = t2 * t2, t1 * t1
        v = (t2s * t2 - 4 * t1s * t1) * (1 / 6)
        dv = (2 * t1s - 0.5 * t2s) * (np.sign(z) / self.h)
        return v, dv


class Model:
    """Sequential MLP or KAN with a flat list of parameter arrays."""

    def __init__(self, kind, dims, terms=8, rng=None, dtype=np.float32):
        self.kind, self.dims, self.terms, self.dtype = kind, list(dims), terms, dtype
        rng = rng if rng is not None else np.random.default_rng(0)
        self.params, self.names = [], []
        if kind.startswith("mlp-"):
            self.act = kind[4:]
            for a, b in zip(dims[:-1], dims[1:]):
                bound = 1 / math.sqrt(a)  # torch.nn.Linear default init
                self.params += [rng.uniform(-bound, bound, (b, a)), rng.uniform(-bound, bound, b)]
                self.names += ["weight", "bias"]
        else:
            self.basis = Basis(kind[4:], terms, dtype)
            for a, b in zip(dims[:-1], dims[1:]):
                self.params += [rng.normal(0, 0.5 / math.sqrt(a), (b, a)),            # base
                                np.ones((b, a)),                                       # scale
                                rng.normal(0, 1, (b, a, terms)) * 0.03 / math.sqrt(a), # coeff
                                np.zeros(b)]                                           # bias
                self.names += ["base", "scale", "coeff", "bias"]
        self.params = [p.astype(dtype) for p in self.params]
        assert self.n_parameters() == parameter_count(kind, dims, terms)

    def n_parameters(self):
        return int(sum(p.size for p in self.params))

    # ---------------------------------------------------------------- forward
    def forward(self, x, keep=False):
        cache = []
        h = x.astype(self.dtype)
        n = len(self.dims) - 1
        for l in range(n):
            if self.kind.startswith("mlp-"):
                W, b = self.params[2 * l: 2 * l + 2]
                z = h @ W.T + b
                if l < n - 1:
                    a, da = _act(self.act, z)
                    cache.append((h, da)); h = a
                else:
                    cache.append((h, None)); h = z
            else:
                base, scale, coeff, bias = self.params[4 * l: 4 * l + 4]
                s, ds = silu(h)
                B, dB = self.basis(h)
                bsz, i = h.shape; o = base.shape[0]; R = self.terms
                Wc = (coeff * scale[..., None]).reshape(o, i * R)
                out = s @ base.T + B.reshape(bsz, i * R) @ Wc.T + bias
                cache.append((s, ds, B, dB))
                h = out
        return (h, cache) if keep else h

    __call__ = forward

    # --------------------------------------------------------------- backward
    def backward(self, cache, dout):
        grads = [None] * len(self.params)
        g = dout
        n = len(self.dims) - 1
        for l in reversed(range(n)):
            if self.kind.startswith("mlp-"):
                W = self.params[2 * l]
                h, da = cache[l]
                if da is not None:
                    g = g * da
                grads[2 * l] = g.T @ h
                grads[2 * l + 1] = g.sum(0)
                g = g @ W
            else:
                base, scale, coeff, bias = self.params[4 * l: 4 * l + 4]
                s, ds, B, dB = cache[l]
                bsz, i = s.shape; o = base.shape[0]; R = self.terms
                grads[4 * l] = g.T @ s
                dWc = (g.T @ B.reshape(bsz, i * R)).reshape(o, i, R)
                grads[4 * l + 1] = (dWc * coeff).sum(-1)
                grads[4 * l + 2] = dWc * scale[..., None]
                grads[4 * l + 3] = g.sum(0)
                Wc = (coeff * scale[..., None]).reshape(o, i * R)
                dBv = (g @ Wc).reshape(bsz, i, R)
                g = (g @ base) * ds + (dBv * dB).sum(-1)
        return grads

    def state(self):
        return [p.copy() for p in self.params]

    def load(self, state):
        self.params = [p.copy() for p in state]


# ------------------------------------------------------------------- losses
def loss_and_grad(pred, y, task):
    if task == "classification":
        z = pred - pred.max(1, keepdims=True)
        p = np.exp(z); p /= p.sum(1, keepdims=True)
        n = len(y)
        loss = -np.log(np.clip(p[np.arange(n), y], 1e-12, None)).mean()
        g = p.copy(); g[np.arange(n), y] -= 1; g /= n
        return float(loss), g.astype(pred.dtype)
    d = pred - y
    return float((d * d).mean()), (2 * d / d.size).astype(pred.dtype)


def loss_only(pred, y, task):
    return loss_and_grad(pred, y, task)[0]


# --------------------------------------------------------------- optimiser
class Adam:
    """torch.optim.Adam semantics (coupled L2 weight decay, bias correction)."""

    def __init__(self, params, lr, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8):
        self.lr, self.wd, self.b1, self.b2, self.eps = lr, weight_decay, betas[0], betas[1], eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        c1, c2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for p, g, m, v in zip(params, grads, self.m, self.v):
            if self.wd:
                g = g + self.wd * p
            m *= self.b1; m += (1 - self.b1) * g
            v *= self.b2; v += (1 - self.b2) * g * g
            p -= self.lr * (m / c1) / (np.sqrt(v / c2) + self.eps)


class Plateau:
    """torch ReduceLROnPlateau(mode='min', threshold_mode='rel')."""

    def __init__(self, factor, patience, threshold):
        self.factor, self.patience, self.threshold = factor, patience, threshold
        self.best, self.bad = math.inf, 0

    def step(self, value, opt):
        if value < self.best * (1 - self.threshold):
            self.best, self.bad = value, 0
        else:
            self.bad += 1
        if self.bad > self.patience:
            opt.lr *= self.factor; self.bad = 0


def clip_(grads, max_norm=10.0):
    total = math.sqrt(sum(float((g.astype(np.float64) ** 2).sum()) for g in grads))
    if total > max_norm:
        f = max_norm / (total + 1e-6)
        for g in grads:
            g *= f
    return total


# ----------------------------------------------------------------- training
def fit(kind, dims, data, cfg, seed, lr):
    """Mirror of study.fit: validation-only stopping and checkpoint selection."""
    rng = np.random.default_rng(seed)
    model = Model(kind, dims, cfg["terms"], rng)
    opt = Adam(model.params, lr, cfg["weight_decay"])
    sched = Plateau(0.5, cfg["lr_patience"], cfg["min_delta"])
    tx, ty, _ = data["train"]; vx, vy, _ = data["val"]
    best = reference = math.inf; stale = 0; history = []; checkpoint = None; best_epoch = 0
    start = time.perf_counter(); reason = "epoch_limit"
    for epoch in range(1, cfg["max_epochs"] + 1):
        pred, cache = model.forward(tx, keep=True)
        loss, g = loss_and_grad(pred, ty, data["task"])
        if not np.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        grads = model.backward(cache, g)
        clip_(grads, 10.0)
        opt.step(model.params, grads)
        if epoch % cfg["check_every"] and epoch != cfg["max_epochs"]:
            continue
        val = loss_only(model(vx), vy, data["task"])
        if not np.isfinite(val):
            raise FloatingPointError("Nonfinite validation loss")
        sched.step(val, opt)
        history.append({"epoch": epoch, "seconds": time.perf_counter() - start,
                        "train_loss": loss, "val_loss": val, "lr": opt.lr})
        if val < best:
            best, checkpoint, best_epoch = val, model.state(), epoch
        if val < reference * (1 - cfg["min_delta"]):
            reference, stale = val, 0
        else:
            stale += 1
        if stale >= cfg["patience"]:
            reason = "validation_plateau"; break
    seconds = time.perf_counter() - start
    model.load(checkpoint)
    return model, {"val_loss": best, "epochs": epoch, "best_epoch": best_epoch, "stop_reason": reason,
                   "fit_seconds": seconds, "lr": lr, "history": history}
