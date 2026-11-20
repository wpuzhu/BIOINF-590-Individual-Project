"""Aggregating spatial-domain results: the paper's baseline and the proposed alternatives.

Baseline (Kang et al. 2025): align every method's labels to a reference by
majority mapping and take the per-spot majority vote. Every method counts
equally, and the output is a hard label with no confidence.

Proposed:
  1. Co-association consensus (CAC). Pool *members*: each (method, perturbation
     replicate) labelling. C[i, j] = weighted fraction of members that put
     spots i and j in the same domain. The consensus partition is obtained by
     clustering C. Each spot gets a soft membership (its mean co-association
     with every consensus domain) and a confidence (max membership). No
     label alignment is needed.
  2. Label-free member weights. Each method is scored without ground truth on:
       stability   - mean ARI between its own replicates (seeds/perturbations)
       centrality  - mean ARI to the other methods' members (agreement)
       coherence   - fraction of spatial neighbours sharing its label
     The weight is w_m = stability_m * centrality_m.
  3. Ensemble pruning. Methods are ranked by score = z(stability) + z(centrality)
     + z(coherence). Methods more than `mad_k` MADs below the median are pruned;
     alternatively the top-q are kept. Pruning targets methods that are both
     unstable and idiosyncratic, which are the ones that drag a vote down.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.neighbors import kneighbors_graph


# ------------------------------------------------------------------ baseline
def align(ref, lab, k):
    """Map labels of `lab` onto `ref` by Hungarian matching of the contingency table."""
    M = np.zeros((k, k))
    np.add.at(M, (lab, ref), 1)
    r, c = linear_sum_assignment(-M)
    mapping = np.arange(k); mapping[r] = c
    return mapping[lab]


def majority_vote(labelings, k, ref_index=0):
    ref = labelings[ref_index]
    aligned = np.array([align(ref, l, k) for l in labelings])
    votes = np.zeros((aligned.shape[1], k))
    for a in aligned:
        votes[np.arange(len(a)), a] += 1
    return votes.argmax(1), votes.max(1) / len(labelings)


# ------------------------------------------------------- proposed: co-association
def coassociation(labelings, weights=None):
    L = np.asarray(labelings)
    w = np.ones(len(L)) if weights is None else np.asarray(weights, float)
    n = L.shape[1]
    C = np.zeros((n, n), dtype=np.float32)
    for l, wi in zip(L, w):
        onehot = np.zeros((n, l.max() + 1), dtype=np.float32); onehot[np.arange(n), l] = 1
        C += wi * (onehot @ onehot.T)
    return C / w.sum()


def consensus_from_coassociation(C, k, seed=0):
    labels = SpectralClustering(k, affinity="precomputed", random_state=seed, assign_labels="cluster_qr").fit_predict(C.astype(float) + 1e-6)
    soft = np.stack([C[:, labels == d].mean(1) if (labels == d).any() else np.zeros(len(C)) for d in range(k)], 1)
    soft = soft / np.maximum(soft.sum(1, keepdims=True), 1e-12)
    return labels, soft, soft.max(1)


# ------------------------------------------------------- label-free method scores
def spatial_coherence(labels, coords, k=6):
    G = kneighbors_graph(coords, k, include_self=False).tocoo()
    return float((labels[G.row] == labels[G.col]).mean())


def method_scores(members, coords):
    """members: dict method -> list of label arrays (replicates on the same spots)."""
    names = list(members)
    out = {}
    for m in names:
        reps = members[m]
        stab = np.mean([ari(reps[i], reps[j]) for i in range(len(reps)) for j in range(i + 1, len(reps))]) if len(reps) > 1 else np.nan
        others = [l for o in names if o != m for l in members[o]]
        cent = np.mean([ari(r, o) for r in reps for o in others]) if others else np.nan
        coh = np.mean([spatial_coherence(r, coords) for r in reps])
        out[m] = {"stability": float(stab), "centrality": float(cent), "coherence": float(coh)}
    z = lambda key: _z(np.nan_to_num(np.array([out[m][key] for m in names]), nan=0.0))
    score = z("stability") + z("centrality") + z("coherence")
    for m, s in zip(names, score):
        out[m]["score"] = float(s)
        st = out[m]["stability"]; ce = out[m]["centrality"]
        # undefined components (a single replicate, or a single method) are neutral
        out[m]["weight"] = (1.0 if np.isnan(st) else max(st, 0)) * (1.0 if np.isnan(ce) else max(ce, 0))
    return out


def _z(v):
    sd = v.std()
    return (v - v.mean()) / sd if sd > 0 else np.zeros_like(v)


def prune(scores, mad_k=1.5, keep_top=None):
    names = sorted(scores, key=lambda m: -scores[m]["score"])
    if keep_top is not None:
        return names[:keep_top]
    s = np.array([scores[m]["score"] for m in names])
    med = np.median(s); mad = np.median(np.abs(s - med)) * 1.4826 + 1e-9
    kept = [m for m, v in zip(names, s) if v >= med - mad_k * mad]
    return kept if kept else names[:1]


def ensemble(members, coords, k, mode="cac_weighted_pruned", seed=0, mad_k=1.5, keep_top=None):
    """mode: 'majority' (paper baseline) | 'cac' | 'cac_weighted' | 'cac_weighted_pruned'."""
    names = list(members)
    if mode == "majority":
        labs = [members[m][0] for m in names]  # one run per method, as in the paper
        lab, conf = majority_vote(labs, k)
        soft = None
        return {"labels": lab, "confidence": conf, "soft": soft, "kept": names, "scores": None}
    scores = method_scores(members, coords)
    kept = prune(scores, mad_k, keep_top) if mode.endswith("pruned") else names
    labs, w = [], []
    for m in kept:
        for r in members[m]:
            labs.append(r); w.append(scores[m]["weight"] if "weighted" in mode else 1.0)
    C = coassociation(labs, np.maximum(w, 1e-3))
    lab, soft, conf = consensus_from_coassociation(C, k, seed)
    return {"labels": lab, "confidence": conf, "soft": soft, "kept": kept, "scores": scores}


# ------------------------------------------------- published-style baselines
def _onehot(l):
    H = np.zeros((len(l), l.max() + 1), dtype=np.float32); H[np.arange(len(l)), l] = 1
    return H


def ensdd_weights(labelings, lam=None, iters=50, tol=1e-6):
    """EnSDD-style weights (Li, Tan & Zhang 2024): minimise
    sum_m w_m ||S - A_m||_F^2 + lam * sum_m w_m log w_m over the simplex, with
    A_m the binary co-membership matrix of member m. Alternating closed forms:
    S = sum_m w_m A_m and w_m proportional to exp(-||S - A_m||^2 / lam). Distances
    are computed through one-hot factors, so no n x n matrix per member is stored.
    lam defaults to the median initial distance (scale-free)."""
    Hs = [_onehot(np.asarray(l)) for l in labelings]
    M = len(Hs)
    G = np.array([[float(((Hi.T @ Hj) ** 2).sum()) for Hj in Hs] for Hi in Hs])  # <A_i, A_j>
    w = np.ones(M) / M
    lam_v = lam
    for _ in range(iters):
        # ||S - A_m||^2 = w'Gw - 2 (Gw)_m + G_mm
        Gw = G @ w
        d = w @ Gw - 2 * Gw + np.diag(G)
        if lam_v is None:
            lam_v = float(np.median(d)) + 1e-9
        new = np.exp(-(d - d.min()) / lam_v); new /= new.sum()
        if np.abs(new - w).max() < tol:
            w = new; break
        w = new
    return w


def stcc_average(labelings, k, seed=0):
    """STCC 'average-based' consensus: k-means on the rows of the unweighted
    co-association matrix."""
    from sklearn.cluster import KMeans
    C = coassociation(labelings)
    return KMeans(k, n_init=5, random_state=seed).fit_predict(C)


def prune_gap(scores, delta=1.0, min_keep=2, max_keep=3):
    """Pre-registered pruning rule (fixed on the pilot simulations before any
    held-out or real data): keep methods whose score is within `delta` (z units)
    of the best, at least `min_keep` and at most `max_keep`."""
    names = sorted(scores, key=lambda m: -scores[m]["score"])
    best = scores[names[0]]["score"]
    kept = [m for m in names if scores[m]["score"] >= best - delta]
    kept = kept[:max_keep]
    if len(kept) < min_keep:
        kept = names[:min_keep]
    return kept


def run_all_ensembles(members, coords, k, seed=0):
    """All aggregation strategies on the same members. Returns dict name -> result."""
    names = list(members)
    out = {}
    out["majority (Kang 2025)"] = ensemble(members, coords, k, "majority", seed=seed)
    flat = [l for m in names for l in members[m]]
    out["STCC-average"] = {"labels": stcc_average(flat, k, seed), "confidence": None, "kept": names}
    owner = [m for m in names for _ in members[m]]
    w = ensdd_weights(flat)
    C = coassociation(flat, w)
    lab, soft, conf = consensus_from_coassociation(C, k, seed)
    wm = {m: float(sum(wi for wi, o in zip(w, owner) if o == m)) for m in names}
    out["EnSDD-weights"] = {"labels": lab, "confidence": conf, "soft": soft, "kept": names, "weights": wm}
    out["CAC weighted"] = ensemble(members, coords, k, "cac_weighted", seed=seed)
    out["PCE (proposed)"] = pce(members, coords, k, seed)
    return out


def pce(members, coords, k, seed=0, prune=True):
    """Pruned, confidence-aware co-association ensemble (proposed): label-free
    scores -> pre-registered gap pruning -> weighted co-association consensus
    -> soft memberships and per-spot confidence."""
    scores = method_scores(members, coords) if len(members) > 1 else {m: {"score": 0.0, "weight": 1.0} for m in members}
    kept = prune_gap(scores) if (prune and len(members) > 2) else list(members)
    res = ensemble({m: members[m] for m in kept}, coords, k, "cac_weighted", seed=seed)
    res["kept"] = kept; res["scores"] = scores
    return res
