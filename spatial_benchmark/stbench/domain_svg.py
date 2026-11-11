"""Domain-specific SVG detection v2: split, core spots, depth-aware rule, and
negative-control-calibrated stability selection ("SplitStab").

Each step addresses a gap left by the reference benchmark and recent work:

1. Double dipping (SpaGCN rule, Kang et al. 2025; also EnSDD/SpatialESD marker
   steps): domains and tests use the same counts. Here domains come from X1
   and tests use X2 (count splitting).
2. Domain uncertainty is ignored downstream (InSTability shows it propagates
   into DE). Here only *core* spots (consensus confidence >= c) are compared,
   so ambiguous boundary spots neither dilute nor fake a contrast.
3. The fixed in-domain detection fraction (>= 0.8) collapses at low depth
   (our pilot). Here the rule is relative: fold change and a detection-rate
   *difference* between inside and outside.
4. Thresholds are tuned by eye and lists carry no error rate. Here the whole
   pipeline is re-run on B replicates (new split + spot subsample + seeds).
   Each gene gets a selection probability pi_g. Synthetic NB negative-control
   genes (real means, dispersions and per-spot size factors, no biology) are
   appended, in the spirit of Castl's controls and ClusterDE's synthetic null. Their
   selection rate estimates the false-discovery proportion at each pi
   threshold, and the threshold is the smallest pi whose estimated FDP <= alpha.
5. Clusters can track non-spatial cell states, so a gene can truly differ
   between clusters without being spatial. A spatial gate (Moran's I + SPARK-X-
   like, ACAT, BH on the held-out half) is required for a domain-SVG call.
"""
import numpy as np
from scipy import stats

from .methods import bh, lognorm


def test_domains(counts_test, labels, confidence=None, core=0.6, q_thr=0.05, fold=1.5, det_diff=0.1,
                 min_spots=10, method="nb_offset", min_rate_quantile=0.1):
    """Return (hit [genes x domains] bool, q [genes x domains], log2 rate ratio [genes x domains]).

    method="nb_offset" (default): depth-adjusted rate test. For each gene and
      domain, rates r = sum(counts) / sum(size factors) inside and outside;
      log(r_in / r_out) has approximate NB variance
      1/(mu S) + sum(sf^2) / (b S^2) per group (mu = rate, S = sum sf, b = NB size).
      One-sided Wald z, BH per domain, call if q < q_thr and r_in / r_out >= fold.
      Size factors come from the test matrix itself, so domains that differ in
      sequencing depth do not produce markers; this is the confound that count
      splitting alone cannot remove.
    method="wilcoxon": v1 rule (log-normalised Mann-Whitney + fold + detection-rate
      difference), kept for comparison; it is NOT depth-robust."""
    if method == "nb_offset":
        return _test_nb_offset(counts_test, labels, confidence, core, q_thr, fold, min_spots, min_rate_quantile)
    x = lognorm(counts_test)
    det = counts_test > 0
    lab = np.asarray(labels)
    keep = np.ones(len(lab), bool) if confidence is None else (np.asarray(confidence) >= core)
    doms = np.unique(lab)
    G = x.shape[1]
    hit = np.zeros((G, len(doms)), bool); Q = np.ones((G, len(doms))); LFC = np.zeros((G, len(doms)))
    for j, d in enumerate(doms):
        inside = keep & (lab == d); outside = keep & (lab != d)
        if inside.sum() < min_spots or outside.sum() < min_spots:
            continue
        p = stats.mannwhitneyu(x[inside], x[outside], alternative="greater", axis=0).pvalue
        q = bh(p)
        m_in = np.expm1(x[inside]).mean(0); m_out = np.expm1(x[outside]).mean(0)
        f_in = det[inside].mean(0); f_out = det[outside].mean(0)
        lfc = np.log2((m_in + 1e-3) / (m_out + 1e-3))
        hit[:, j] = (q < q_thr) & (m_in >= fold * np.maximum(m_out, 1e-8)) & (f_in - f_out >= det_diff)
        Q[:, j] = q; LFC[:, j] = lfc
    return hit, Q, LFC


def _test_nb_offset(X, labels, confidence, core, q_thr, fold, min_spots, min_rate_quantile):
    from .splitting import estimate_nb_size
    lab = np.asarray(labels)
    keep = np.ones(len(lab), bool) if confidence is None else (np.asarray(confidence) >= core)
    lib = X.sum(1).astype(float); sf = lib / np.median(lib[lib > 0])
    b = estimate_nb_size(X)
    doms = np.unique(lab); G = X.shape[1]
    overall = X.sum(0) / sf.sum()
    expressed = overall >= np.quantile(overall, min_rate_quantile)
    hit = np.zeros((G, len(doms)), bool); Q = np.ones((G, len(doms))); LFC = np.zeros((G, len(doms)))
    for j, d in enumerate(doms):
        inside = keep & (lab == d); outside = keep & (lab != d)
        if inside.sum() < min_spots or outside.sum() < min_spots:
            continue
        Oi, Oo = X[inside].sum(0).astype(float), X[outside].sum(0).astype(float)
        Si, So = sf[inside].sum(), sf[outside].sum()
        S2i, S2o = (sf[inside] ** 2).sum(), (sf[outside] ** 2).sum()
        ri, ro = (Oi + 0.5) / Si, (Oo + 0.5) / So
        vi = 1 / (ri * Si) + S2i / (b * Si * Si); vo = 1 / (ro * So) + S2o / (b * So * So)
        z = (np.log(ri) - np.log(ro)) / np.sqrt(vi + vo)
        p = stats.norm.sf(z); q = bh(p)
        rr = ri / ro
        hit[:, j] = (q < q_thr) & (rr >= fold) & expressed
        Q[:, j] = q; LFC[:, j] = np.log2(rr)
    return hit, Q, LFC


def spatial_gate(counts_test, coords, q_thr=0.05):
    """Gene-level spatial test on the held-out counts: ACAT of Moran's I and the
    SPARK-X-style kernel test, BH-adjusted. A *domain-specific SVG* must be
    enriched in a domain AND spatially variable; without this gate, clusters that
    track non-spatial cell states turn cell-state genes into 'domain markers'
    even after count splitting (they really differ between those clusters)."""
    from .methods import acat, knn_weights, morans_i, sparkx_like
    x = lognorm(counts_test)
    _, pI = morans_i(x, knn_weights(coords))
    pS = sparkx_like(x, coords)
    q = bh(acat(np.vstack([pI, pS]), axis=0))
    return q < q_thr, q


def add_negative_controls(counts, frac=0.1, rng=None, kind="nb"):
    """Append negative-control genes. Returns (augmented, n_nc).

    kind="nb" (default): synthetic genes drawn as NB(mu_g * sf_i, b_g) with the
      means/dispersions of randomly chosen real genes and the REAL per-spot size
      factors. They carry the slice's technical depth structure but no biology,
      so they are selected exactly as often as depth artefacts would be (cf. the
      synthetic null of ClusterDE).
    kind="permute": permuted copies of real genes (Castl-style). These also
      destroy the depth association, so they under-estimate depth-driven false
      positives; kept for comparison."""
    rng = rng or np.random.default_rng(0)
    G = counts.shape[1]; n_nc = max(1, int(round(frac * G)))
    src = rng.choice(G, n_nc, replace=False)
    if kind == "permute":
        nc = np.stack([rng.permutation(counts[:, g]) for g in src], 1)
    else:
        from .splitting import estimate_nb_size
        lib = counts.sum(1).astype(float); sf = lib / np.median(lib[lib > 0])
        mu = (counts[:, src] / np.maximum(sf, 1e-8)[:, None]).mean(0)
        b = estimate_nb_size(counts[:, src], sf=sf)
        m = sf[:, None] * mu[None, :]
        nc = rng.negative_binomial(np.broadcast_to(b, m.shape), b[None, :] / (b[None, :] + m))
    return np.concatenate([counts, nc.astype(counts.dtype)], 1), n_nc


def nc_calibrated_threshold(pi_real, pi_nc, alpha=0.1, grid=None):
    """Smallest pi threshold with estimated FDP <= alpha, where
    FDP_hat(t) = (#NC with pi >= t) * (n_real / n_nc) / max(1, #real with pi >= t)."""
    grid = np.round(np.arange(0.5, 1.0001, 0.05), 2) if grid is None else grid
    scale = len(pi_real) / max(len(pi_nc), 1)
    table = []
    chosen = None
    for t in grid:
        n_r = int((pi_real >= t).sum()); n_n = int((pi_nc >= t).sum())
        fdp = n_n * scale / max(n_r, 1)
        table.append({"pi_thr": float(t), "n_selected": n_r, "n_nc_selected": n_n, "fdp_hat": float(fdp)})
        if chosen is None and fdp <= alpha:
            chosen = float(t)
    return (chosen if chosen is not None else 1.0), table


def mb_bound(q, p, pi_thr):
    """Meinshausen-Buhlmann / CPSS basic bound on E[false selections]."""
    return float(q * q / ((2 * pi_thr - 1) * p)) if pi_thr > 0.5 else float("inf")
