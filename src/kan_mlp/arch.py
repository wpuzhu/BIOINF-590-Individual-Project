"""Torch-free architecture search and capacity accounting.

Both the PyTorch implementation (models.py) and the NumPy engine (engine_np.py)
build exactly the same parameter tensors, so capacity matching lives here.

Parameter counts (per layer a -> b):
  MLP linear layer : a*b + b
  KAN layer        : (R+2)*a*b + b   (base a*b, scale a*b, coeff a*b*R, bias b)
"""
from functools import lru_cache
import math

MODELS = ["mlp-tanh", "mlp-relu", "mlp-silu", "kan-spline", "kan-fourier", "kan-rbf"]


def parameter_count(kind, dims, terms=8):
    factor = 1 if kind.startswith("mlp-") else terms + 2
    return sum(factor * a * b + b for a, b in zip(dims[:-1], dims[1:]))


def macs_per_sample(kind, dims, terms=8):
    """Multiply-accumulate operations in the dense forward pass (basis evaluation
    counted as one operation per basis value). Equal parameters do not imply
    equal compute, so both are reported."""
    if kind.startswith("mlp-"):
        return sum(a * b for a, b in zip(dims[:-1], dims[1:]))
    return sum(a * b * (terms + 1) + a * terms for a, b in zip(dims[:-1], dims[1:]))


@lru_cache(maxsize=4096)
def architecture(kind, inputs, outputs, depth, budget, terms=8, max_width=256):
    """Return (dims, parameters) closest to `budget`.

    depth = number of hidden layers. Hidden layers share width w except the last,
    which may take any width in [w/2, 2w] so that the count can be matched
    closely. No dummy or frozen parameters are added.
    """
    best = None
    factor = 1 if kind.startswith("mlp-") else terms + 2
    limit = max(max_width, math.ceil((budget - outputs) / (factor * (inputs + outputs) + 1)) + 2) if depth == 1 else max_width
    for w in range(1, limit + 1):
        tails = [w] if depth == 1 else range(max(1, w // 2), min(max_width, 2 * w) + 1)
        for tail in tails:
            hidden = [w] * (depth - 1) + [tail]
            dims = [inputs] + hidden + [outputs]
            p = parameter_count(kind, dims, terms)
            key = (abs(p - budget), max(hidden) / min(hidden), p)
            if best is None or key < best[0]:
                best = (key, dims, p)
    return best[1], best[2]
