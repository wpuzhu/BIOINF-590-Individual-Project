"""Count splitting to remove double dipping in domain -> marker-gene pipelines.

Problem (Song et al., ClusterDE; Neufeld et al. 2023): if domains are found from
the counts X and the same X is then tested for genes that differ between those
domains, the test is biased. Clustering already pushed apart the spots that
differ, so even genes with no spatial structure look like domain markers.

Count splitting creates two *independent* matrices from one:
  Poisson X:           X1 | X ~ Binomial(X, eps),                     X2 = X - X1
  NB(mu, size b) X:    X1 | X ~ BetaBinomial(X, eps*b, (1-eps)*b),    X2 = X - X1
(Neufeld, Dharamshi, Gao, Witten 2023; Neufeld et al. 2024 for the NB case).
Domains are found on X1 and markers tested on X2, so the test sees data the
clustering never touched. The NB version needs the gene-wise size b; it is
estimated by moments on size-factor-normalised counts and capped, which is the
usual practical choice.
"""
import numpy as np


def estimate_nb_size(counts, cap=1e4, sf=None):
    """Gene-wise NB size by moments on size-factor-normalised counts. Pass `sf`
    when `counts` is a subset of genes (size factors must come from all genes)."""
    if sf is None:
        lib = counts.sum(1).astype(float)
        sf = lib / np.median(lib[lib > 0])
    y = counts / np.maximum(sf, 1e-8)[:, None]
    mu = y.mean(0); var = y.var(0)
    excess = var - mu
    size = np.where(excess > 1e-8, mu ** 2 / np.maximum(excess, 1e-12), cap)
    return np.clip(size, 0.05, cap)


def count_split(counts, eps=0.5, rng=None, model="nb", size=None):
    rng = rng or np.random.default_rng(0)
    X = np.asarray(counts)
    if model == "poisson":
        X1 = rng.binomial(X, eps)
    else:
        b = estimate_nb_size(X) if size is None else size
        a1 = np.broadcast_to(eps * b, X.shape); a2 = np.broadcast_to((1 - eps) * b, X.shape)
        p = rng.beta(a1, a2)
        X1 = rng.binomial(X, p)
    X1 = X1.astype(X.dtype)
    return X1, X - X1
