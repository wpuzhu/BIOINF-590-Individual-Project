"""Base methods: preprocessing, spatial-domain callers, SVG tests.

These are deliberately lightweight stand-ins (no GPU, no GNN libraries) that
span the same design space as the methods in Kang et al. (2025):
  expression-only clustering   -> kmeans_pca, gmm_pca          (Seurat/Leiden-like, group GE)
  expression + space           -> kmeans_smooth, ward_spatial,
                                  spectral_spatial             (BayesSpace/GNN-like, group GE+XY)
Global SVG tests: Moran's I, Geary's C, a SPARK-X-style kernel covariance
test, and a highly-variable-gene (HVG) baseline. Domain-specific SVGs follow
the SpaGCN rule used in the paper (p < 0.05, in-fraction, fold change).
"""
import numpy as np
from scipy import sparse, stats
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors, kneighbors_graph


# ------------------------------------------------------------------ preprocessing
def lognorm(counts, target=1e4):
    lib = counts.sum(1, keepdims=True).astype(float)
    return np.log1p(counts / np.maximum(lib, 1) * target)


def hvg(x, n=500):
    m = x.mean(0); v = x.var(0)
    disp = v / np.maximum(m, 1e-8)
    keep = np.where(m > 0.05)[0]
    return keep[np.argsort(-disp[keep])[:n]]


def knn_weights(coords, k=6):
    """Row-standardised kNN spatial weights (sparse)."""
    W = kneighbors_graph(coords, k, mode="connectivity", include_self=False)
    W = ((W + W.T) > 0).astype(float)
    d = np.asarray(W.sum(1)).ravel()
    return sparse.diags(1 / np.maximum(d, 1)) @ W


def embed(counts, n_pcs=20, n_hvg=500, seed=0):
    x = lognorm(counts)
    g = hvg(x, n_hvg)
    z = (x[:, g] - x[:, g].mean(0)) / (x[:, g].std(0) + 1e-8)
    return PCA(n_pcs, random_state=seed).fit_transform(z)


# ---------------------------------------------------------------- domain callers
def kmeans_pca(pcs, coords, k, seed):
    return KMeans(k, n_init=5, random_state=seed).fit_predict(pcs)


def gmm_pca(pcs, coords, k, seed):
    return GaussianMixture(k, covariance_type="diag", n_init=2, random_state=seed).fit_predict(pcs[:, :10])


def kmeans_smooth(pcs, coords, k, seed, steps=2):
    """Neighbour-averaged embedding (spatial smoothing), then k-means."""
    W = knn_weights(coords, 6)
    z = pcs.copy()
    for _ in range(steps):
        z = 0.5 * z + 0.5 * (W @ z)
    return KMeans(k, n_init=5, random_state=seed).fit_predict(z)


def ward_spatial(pcs, coords, k, seed):
    """Ward clustering constrained to spatially contiguous merges."""
    rng = np.random.default_rng(seed)
    conn = kneighbors_graph(coords, 6, include_self=False)
    z = pcs[:, :15] + rng.normal(0, 1e-6, pcs[:, :15].shape)  # seed only breaks ties
    return AgglomerativeClustering(k, linkage="ward", connectivity=conn).fit_predict(z)


def spectral_spatial(pcs, coords, k, seed, alpha=0.5):
    """Spectral clustering on a graph mixing expression and spatial neighbours."""
    n = len(pcs)
    Ge = kneighbors_graph(pcs[:, :15], 15, include_self=False)
    Gs = kneighbors_graph(coords, 6, include_self=False)
    A = alpha * (Ge + Ge.T) / 2 + (1 - alpha) * (Gs + Gs.T) / 2
    return SpectralClustering(k, affinity="precomputed", random_state=seed, assign_labels="cluster_qr").fit_predict(A)


DOMAIN_METHODS = {
    "kmeans_pca": kmeans_pca, "gmm_pca": gmm_pca, "kmeans_smooth": kmeans_smooth,
    "ward_spatial": ward_spatial, "spectral_spatial": spectral_spatial,
}


# --------------------------------------------------------------- global SVG tests
def morans_i(x, W):
    """Moran's I per gene with normality-assumption z-test (one-sided, positive autocorrelation)."""
    n = x.shape[0]
    z = x - x.mean(0)
    num = (z * (W @ z)).sum(0)
    den = (z * z).sum(0)
    S0 = W.sum(); Wt = W + W.T
    S1 = 0.5 * Wt.multiply(Wt).sum(); S2 = (np.asarray(W.sum(1)).ravel() + np.asarray(W.sum(0)).ravel()) ** 2
    S2 = S2.sum()
    I = n / S0 * num / np.maximum(den, 1e-12)
    EI = -1 / (n - 1)
    VI = (n * n * S1 - n * S2 + 3 * S0 * S0) / ((n * n - 1) * S0 * S0) - EI ** 2
    zscore = (I - EI) / np.sqrt(VI)
    return I, stats.norm.sf(zscore)


def gearys_c(x, W):
    n = x.shape[0]
    z = x - x.mean(0)
    Wc = W.tocoo()
    d = x[Wc.row] - x[Wc.col]
    diff2 = (Wc.data[:, None] * d * d).sum(0)
    S0 = W.sum(); Wt = W + W.T
    S1 = 0.5 * Wt.multiply(Wt).sum()
    S2 = ((np.asarray(W.sum(1)).ravel() + np.asarray(W.sum(0)).ravel()) ** 2).sum()
    C = (n - 1) * diff2 / (2 * S0 * np.maximum((z * z).sum(0), 1e-12))
    VC = ((2 * S1 + S2) * (n - 1) - 4 * S0 * S0) / (2 * (n + 1) * S0 * S0)
    return C, stats.norm.cdf((C - 1) / np.sqrt(VC))  # small C = positive autocorrelation


def _satterthwaite_p(stat, lam):
    """P(sum lam_k chi2_1 > stat) via scaled chi-square matching mean and variance."""
    m, v = lam.sum(), 2 * (lam ** 2).sum()
    scale, df = v / (2 * m), 2 * m * m / v
    return stats.chi2.sf(stat / scale, df)


def acat(pvals, axis=0, weights=None):
    """Cauchy combination test (Liu & Xie 2020): valid under arbitrary dependence."""
    p = np.clip(np.asarray(pvals, float), 1e-15, 1 - 1e-15)
    w = np.ones(p.shape[axis]) / p.shape[axis] if weights is None else np.asarray(weights) / np.sum(weights)
    shape = [1] * p.ndim; shape[axis] = -1
    t = (w.reshape(shape) * np.tan((0.5 - p) * np.pi)).sum(axis)
    return stats.cauchy.sf(t)


def sparkx_like(x, coords):
    """SPARK-X-style covariance test: y'EE'y / y'y for 11 coordinate transforms
    (raw, 5 Gaussian, 5 cosine), each with a mixture-of-chi-square null, combined by ACAT."""
    n = len(x)
    y = x - x.mean(0); y = y / np.maximum(np.sqrt((y * y).mean(0)), 1e-12)
    c = (coords - coords.mean(0)) / coords.std(0)
    transforms = [c]
    for q in np.quantile(np.abs(c), [0.2, 0.4, 0.6, 0.8, 1.0]):
        transforms.append(np.exp(-c ** 2 / (2 * q * q)))
        transforms.append(np.cos(2 * np.pi * c / q))
    ps = []
    for E in transforms:
        E = E - E.mean(0)
        E = E / np.maximum(np.sqrt((E * E).sum(0)), 1e-12)
        lam = np.linalg.eigvalsh(E.T @ E)
        stat = ((E.T @ y) ** 2).sum(0)  # ~ sum lam chi2_1 under spatial exchangeability
        ps.append(_satterthwaite_p(stat, np.clip(lam, 0, None)))
    return acat(np.array(ps), axis=0)


def hvg_score(x):
    m = x.mean(0); v = x.var(0)
    return v / np.maximum(m, 1e-8)


def global_svg_tests(counts, coords, k=6):
    """Return dict test -> (score where larger = more spatial, p-value or None)."""
    x = lognorm(counts)
    W = knn_weights(coords, k)
    I, pI = morans_i(x, W)
    C, pC = gearys_c(x, W)
    pS = sparkx_like(x, coords)
    return {"morans_i": (I, pI), "gearys_c": (-C, pC), "sparkx_like": (-np.log10(np.clip(pS, 1e-300, 1)), pS),
            "hvg": (hvg_score(x), None)}


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.empty(n); q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)


# ---------------------------------------------------- domain-specific SVG (SpaGCN rule)
def domain_svgs(counts, labels, p_thr=0.05, in_frac=0.8, fold=1.5, out_ratio=1.0, weights=None):
    """SpaGCN-style rule per (gene, domain): Wilcoxon p < p_thr (BH), in-domain
    detection fraction >= in_frac, in/out mean fold-change >= fold, and in/out
    detection-fraction ratio >= out_ratio. Returns boolean matrix genes x domains
    and the p-values. If `weights` (spots x domains soft memberships) is given,
    means and fractions are membership-weighted (soft domains)."""
    x = lognorm(counts)
    detected = counts > 0
    doms = np.unique(labels) if weights is None else np.arange(weights.shape[1])
    G = x.shape[1]
    hit = np.zeros((G, len(doms)), bool); pmat = np.ones((G, len(doms)))
    for j, d in enumerate(doms):
        w_in = (labels == d).astype(float) if weights is None else weights[:, j]
        w_out = 1 - w_in
        if w_in.sum() < 5 or w_out.sum() < 5:
            continue
        hard = w_in >= 0.5
        if hard.sum() < 5 or (~hard).sum() < 5:
            continue
        p = stats.mannwhitneyu(x[hard], x[~hard], alternative="greater", axis=0).pvalue
        q = bh(p)
        mean_in = (w_in @ np.expm1(x)) / w_in.sum(); mean_out = (w_out @ np.expm1(x)) / w_out.sum()
        f_in = (w_in @ detected) / w_in.sum(); f_out = (w_out @ detected) / w_out.sum()
        hit[:, j] = (q < p_thr) & (f_in >= in_frac) & (mean_in >= fold * np.maximum(mean_out, 1e-8)) & (f_in >= out_ratio * f_out)
        pmat[:, j] = q
    return hit, pmat
