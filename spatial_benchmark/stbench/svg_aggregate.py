"""Aggregating SVG results and quantifying their uncertainty.

Global SVG reference
  intersection  - paper baseline: genes significant in every test (BH q < 0.05),
                  then "rescued" if Moran's I exceeds the mean Moran's I of the
                  intersection set (the paper's refinement step)
  acat          - proposed: Cauchy combination of the per-test p-values
                  (valid under arbitrary dependence between tests), then BH
  rank_agg      - proposed (p-value-free): mean of per-test rank percentiles

Domain-specific SVGs with uncertainty (proposed: "stability-selected SVGs")
  Re-run the whole pipeline (perturb data -> domains -> consensus -> SpaGCN
  rule) on B perturbation replicates. Each gene's selection probability
  pi_g = fraction of replicates in which it is called. Genes with pi_g >= pi_thr
  are reported, together with the Meinshausen-Buhlmann bound on the expected
  number of false selections E[V] <= q^2 / ((2 pi_thr - 1) p), where q is the
  mean number of genes called per replicate and p the number of genes tested.
"""
import numpy as np
from scipy import stats

from .methods import acat, bh


def global_reference(tests, alpha=0.05, use=("morans_i", "gearys_c", "sparkx_like")):
    P = np.array([tests[t][1] for t in use])
    Q = np.array([bh(p) for p in P])
    inter = (Q < alpha).all(0)
    I = tests["morans_i"][0]
    rescue_thr = I[inter].mean() if inter.any() else np.inf
    inter_rescued = inter | (I > rescue_thr)
    pa = acat(P, axis=0)
    ranks = np.mean([stats.rankdata(tests[t][0]) / len(tests[t][0]) for t in use], 0)
    return {"intersection": inter, "intersection_rescued": inter_rescued, "acat_p": pa, "acat": bh(pa) < alpha,
            "rank_score": ranks, "per_test_q": dict(zip(use, Q))}


def precision_recall(called, truth):
    tp = (called & truth).sum(); fp = (called & ~truth).sum(); fn = (~called & truth).sum()
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    return {"n_called": int(called.sum()), "precision": float(prec), "recall": float(rec),
            "fdp": float(fp / max(tp + fp, 1)), "f1": float(2 * prec * rec / (prec + rec)) if tp else 0.0}


def aupr(score, truth):
    o = np.argsort(-score); t = truth[o]
    tp = np.cumsum(t); prec = tp / np.arange(1, len(t) + 1)
    return float((prec * t).sum() / max(t.sum(), 1))


def agreement(call_sets):
    """Agreement score of Kang et al. (eq. 9): mean over genes in the union of (n_g - 1)/(N - 1)."""
    N = len(call_sets)
    union = np.any(call_sets, 0)
    if not union.any() or N < 2:
        return np.nan
    ng = np.sum(call_sets, 0)[union]
    return float(((ng - 1) / (N - 1)).mean())


def stability_selection(call_matrix, pi_thr=0.6):
    """call_matrix: replicates x genes boolean. Returns probabilities, selection and the MB error bound."""
    pi = call_matrix.mean(0)
    q = call_matrix.sum(1).mean(); p = call_matrix.shape[1]
    bound = q * q / ((2 * pi_thr - 1) * p) if pi_thr > 0.5 else np.inf
    return pi, pi >= pi_thr, float(bound)


def jaccard(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0
