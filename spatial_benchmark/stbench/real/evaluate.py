"""Real-data experiments (one function per experiment; each returns plain dicts).

R1  Domains vs manual layers: single methods, majority vote (Kang 2025),
    STCC-average, EnSDD-weights, CAC weighted, PCE (proposed). ARI, NMI,
    confidence calibration.
R2  Double-dipping null: a slice-calibrated negative-binomial null (each gene's
    mean and dispersion and each spot's library size from the real slice, but no
    spatial or domain structure). Every domain-SVG call is false.
R3  Semi-synthetic spike-in: implant known domain markers into low-autocorrelation
    real genes, inside one annotated layer, at fold changes 1.5/2/3. Real noise,
    known truth.
R4  Adjacent-section replication (same donor, 10 um apart): does the SVG list
    replicate? This is a truth-free validity check.
R5  Perturbation robustness on real slices: depth thinning, spot dropout.
R6  Global SVG references: intersection (Kang) vs ACAT vs Castl-style.
"""
import numpy as np
from sklearn.metrics import adjusted_rand_score as ari, normalized_mutual_info_score as nmi

from .. import ensemble as E
from .. import methods as M
from .. import svg_aggregate as A
from .. import workflow as W
from ..domain_svg import spatial_gate, test_domains
from ..splitting import count_split, estimate_nb_size


def _ece(conf, correct, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (conf >= lo) & ((conf < hi) if hi < 1 else (conf <= hi))
        if s.any():
            e += s.mean() * abs(correct[s].mean() - conf[s].mean())
    return float(e)


def r1_domains(ctx, k, methods, n_reps=3, seed=0, log=None):
    truth = ctx.extra["labels"]
    rng = np.random.default_rng(seed)
    members = W.run_members(ctx, k, methods, n_reps, rng, log=log)
    rows = []
    for m, labs in members.items():
        rows.append({"approach": m, "kind": "single", "ari": float(np.mean([ari(truth, l) for l in labs])),
                     "ari_sd": float(np.std([ari(truth, l) for l in labs])), "nmi": float(np.mean([nmi(truth, l) for l in labs]))})
    ens = E.run_all_ensembles(members, ctx.coords, k, seed)
    for name, res in ens.items():
        rec = {"approach": name, "kind": "ensemble", "ari": ari(truth, res["labels"]), "nmi": nmi(truth, res["labels"]),
               "kept": ",".join(res.get("kept") or [])}
        conf = res.get("confidence")
        if conf is not None:
            correct = E.align(truth, res["labels"], max(k, truth.max() + 1)) == truth
            rec.update(ece=_ece(conf, correct), share_core=float((conf >= 0.6).mean()),
                       acc_core=float(correct[conf >= 0.6].mean()) if (conf >= 0.6).any() else np.nan,
                       acc_noncore=float(correct[conf < 0.6].mean()) if (conf < 0.6).any() else np.nan)
        rows.append(rec)
    scores = ens["PCE (proposed)"]["scores"]
    return rows, members, ens, scores


def nb_null(ctx, rng):
    """Counts with the slice's gene means/dispersions and spot library sizes, but no structure."""
    X = ctx.counts
    lib = X.sum(1).astype(float); sf = lib / np.median(lib)
    mu = (X / sf[:, None]).mean(0)
    size = estimate_nb_size(X)
    m = sf[:, None] * mu[None, :]
    p = size[None, :] / (size[None, :] + m)
    return rng.negative_binomial(np.broadcast_to(size, m.shape), p).astype(np.int32)


def r2_null(ctx, k, methods, B=6, seed=0, log=None):
    rng = np.random.default_rng(seed)
    null = W.Context(nb_null(ctx, rng), ctx.coords, ctx.genes, ctx.cluster_genes, {})
    out = {}
    called, _ = W.paper_pipeline(null, k, methods, rng, strict=True); out["paper rule, same data (strict)"] = int(called.sum())
    called, _ = W.paper_pipeline(null, k, methods, rng, strict=False); out["paper rule, same data (liberal)"] = int(called.sum())
    members = W.run_members(null, k, methods, 2, rng, log=log)
    res = E.pce(members, null.coords, k, seed)
    hit, _, _ = test_domains(null.counts, res["labels"], res["confidence"]); out["v2 rule, same data"] = int(hit.any(1).sum())
    X1, X2 = count_split(null.counts, 0.5, rng)
    c1 = W.Context(X1, null.coords, null.genes, null.cluster_genes, {})
    r1 = E.pce(W.run_members(c1, k, methods, 2, rng, log=log), c1.coords, k, seed)
    hit, _, _ = test_domains(X2, r1["labels"], r1["confidence"]); out["v2 rule, count split"] = int(hit.any(1).sum())
    out["v2 rule, count split + spatial gate"] = int((hit.any(1) & spatial_gate(X2, null.coords)[0]).sum())
    out["v1 stability (liberal rule, same data)"] = int(W.v1_stability(null, k, methods, B=B, rng=rng)["selected"].sum())
    ss = W.splitstab(null, k, methods, B=B, rng=rng, log=log)
    out["SplitStab strict (v2)"] = int(ss["by_rule"]["strict"]["selected"].sum())
    out["SplitStab combined (proposed)"] = int(ss["by_rule"]["liberal"]["selected"].sum())
    out["global ACAT split + stability"] = int(ss["global"]["selected"].sum())
    out["n_genes"] = int(ctx.counts.shape[1])
    T = M.global_svg_tests(null.counts, null.coords)
    out["global ACAT (BH<0.05)"] = int(A.global_reference(T)["acat"].sum())
    out["global intersection"] = int(A.global_reference(T)["intersection_rescued"].sum())
    return out


def spike_in(ctx, rng, n_spike=150, folds=(1.5, 2.0, 3.0)):
    """Implant markers into real low-autocorrelation genes inside one annotated layer each."""
    labels = ctx.extra["labels"]
    x = M.lognorm(ctx.counts)
    I, _ = M.morans_i(x, M.knn_weights(ctx.coords))
    mean = ctx.counts.mean(0)
    pool = np.where((np.abs(I) < np.quantile(np.abs(I), 0.3)) & (mean > np.quantile(mean, 0.2)))[0]
    spiked = rng.choice(pool, min(n_spike, len(pool) // 2), replace=False)
    presumed_null = np.setdiff1d(pool, spiked)
    X = ctx.counts.copy()
    lib = X.sum(1).astype(float); sf = lib / np.median(lib)
    layers = np.unique(labels)
    target = rng.choice(layers, len(spiked)); fold = rng.choice(folds, len(spiked))
    for g, d, f in zip(spiked, target, fold):
        inside = labels == d
        mu = (X[:, g] / sf).mean()
        X[inside, g] += rng.poisson((f - 1) * mu * sf[inside]).astype(X.dtype)
    sctx = W.Context(X, ctx.coords, ctx.genes, ctx.cluster_genes, dict(ctx.extra))
    return sctx, {"spiked": spiked, "target": target, "fold": fold, "presumed_null": presumed_null}


def r3_spikein(ctx, k, methods, B=10, seed=0, log=None):
    rng = np.random.default_rng(seed)
    sctx, info = spike_in(ctx, rng)
    rows = []
    def score(name, called, extra=None):
        sp, pn = info["spiked"], info["presumed_null"]
        rec = {"approach": name, "recall_spiked": float(called[sp].mean()),
               "false_call_rate_presumed_null": float(called[pn].mean()), "n_called": int(called.sum())}
        for f in np.unique(info["fold"]):
            rec[f"recall_fold_{f}"] = float(called[sp[info["fold"] == f]].mean())
        rec.update(extra or {})
        rows.append(rec)
    called, _ = W.paper_pipeline(sctx, k, methods, rng, strict=True); score("Kang 2025 pipeline (majority + strict SpaGCN rule)", called)
    called, _ = W.paper_pipeline(sctx, k, methods, rng, strict=False); score("majority + liberal rule, same data", called)
    score("v1 stability (liberal rule, same data)", W.v1_stability(sctx, k, methods, B=B, rng=rng)["selected"])
    ss = W.splitstab(sctx, k, methods, B=B, rng=rng, log=log)
    score("SplitStab strict (v2)", ss["by_rule"]["strict"]["selected"], {"threshold": ss["by_rule"]["strict"]["threshold"]})
    score("SplitStab combined (proposed)", ss["by_rule"]["liberal"]["selected"], {"threshold": ss["by_rule"]["liberal"]["threshold"]})
    return rows, info, ss


def r4_replication(lists_a, lists_b, genes_a, genes_b):
    """lists_*: dict approach -> boolean vector over genes_*. Compare on shared genes."""
    shared = sorted(set(genes_a) & set(genes_b))
    ia = {g: i for i, g in enumerate(genes_a)}; ib = {g: i for i, g in enumerate(genes_b)}
    rows = []
    for name in lists_a:
        if name not in lists_b:
            continue
        a = np.array([lists_a[name][ia[g]] for g in shared]); b = np.array([lists_b[name][ib[g]] for g in shared])
        rows.append({"approach": name, "n_a": int(a.sum()), "n_b": int(b.sum()), "jaccard": A.jaccard(a, b),
                     "replication_rate": float((a & b).sum() / max(a.sum(), 1))})
    return rows


def r5_perturb(ctx, k, methods, base_lists, B=6, seed=0, settings=(("depth", 0.5), ("depth", 0.25), ("spot_frac", 0.6)), log=None):
    """Unperturbed reference lists are recomputed with the SAME methods and B, so
    perturbed-vs-unperturbed differences are due to the perturbation only."""
    rows = []
    truth = ctx.extra.get("labels")
    rng0 = np.random.default_rng(seed)
    base_lists = dict(base_lists)
    base_lists["Kang 2025 pipeline"] = W.paper_pipeline(ctx, k, methods, rng0, strict=True)[0]
    base_lists["SplitStab combined (proposed)"] = W.splitstab(ctx, k, methods, B=B, rng=rng0, log=log)["selected"]
    for name, sel in [("Kang 2025 pipeline", base_lists["Kang 2025 pipeline"]), ("SplitStab combined (proposed)", base_lists["SplitStab combined (proposed)"])]:
        rows.append({"axis": "none", "level": 1.0, "approach": name, "n_called": int(sel.sum()), "jaccard_to_unperturbed": 1.0})
    for axis, level in settings:
        rng = np.random.default_rng(seed + int(level * 100))
        if axis == "depth":
            idx = np.arange(len(ctx.coords)); X = rng.binomial(ctx.counts, level).astype(ctx.counts.dtype)
        else:
            idx = np.sort(rng.choice(len(ctx.coords), int(level * len(ctx.coords)), replace=False)); X = ctx.counts[idx]
        p = W.Context(X, ctx.coords[idx], ctx.genes, ctx.cluster_genes, {})
        # note: for spot dropout the gene universe is unchanged, so lists stay comparable
        called, res = W.paper_pipeline(p, k, methods, rng, strict=True)
        rec = {"axis": axis, "level": level, "approach": "Kang 2025 pipeline", "n_called": int(called.sum()),
               "jaccard_to_unperturbed": A.jaccard(called, base_lists["Kang 2025 pipeline"])}
        if truth is not None:
            rec["ari"] = ari(truth[idx], res["labels"])
        rows.append(rec)
        ss = W.splitstab(p, k, methods, B=B, rng=rng, log=log)
        rows.append({"axis": axis, "level": level, "approach": "SplitStab combined (proposed)", "n_called": int(ss["selected"].sum()),
                     "jaccard_to_unperturbed": A.jaccard(ss["selected"], base_lists["SplitStab combined (proposed)"])})
        T = M.global_svg_tests(p.counts, p.coords); R = A.global_reference(T)
        for name in ["intersection_rescued", "acat"]:
            rows.append({"axis": axis, "level": level, "approach": f"global {name}", "n_called": int(R[name].sum()),
                         "jaccard_to_unperturbed": A.jaccard(R[name], base_lists[f"global {name}"])})
    return rows


def boundary_distance(coords, labels):
    """Distance (spot spacings) from each spot to the nearest spot with a different label."""
    from scipy.spatial import cKDTree
    d = np.full(len(labels), np.inf)
    for l in np.unique(labels):
        other = coords[labels != l]
        if len(other):
            d[labels == l] = cKDTree(other).query(coords[labels == l])[0]
    return d


def spot_uncertainty(ctx, labels, confidence, agreement, k, stability=None):
    """Biological check of spot-level uncertainty on real tissue: is low confidence
    concentrated at manually annotated layer boundaries (where Visium spots mix
    cells from two layers), and are confident spots more often right?"""
    from scipy.stats import spearmanr
    truth = ctx.extra["labels"]
    correct = E.align(truth, labels, max(k, truth.max() + 1)) == truth
    bd = boundary_distance(ctx.coords, truth)
    rows = {"n_spots": len(truth)}
    for name, u in [("confidence", confidence), ("replicate_stability", stability), ("replicate_agreement", agreement)]:
        if u is None:
            continue
        ok = np.isfinite(u)
        rows[f"{name}_mean"] = float(np.nanmean(u))
        rows[f"{name}_spearman_vs_boundary_distance"] = float(spearmanr(u[ok], bd[ok])[0])
        rows[f"{name}_at_boundary(<=1.5)"] = float(np.nanmean(u[bd <= 1.5]))
        rows[f"{name}_interior(>3)"] = float(np.nanmean(u[bd > 3]))
        hi = ok & (u >= 0.6); lo = ok & (u < 0.6)
        rows[f"accuracy_{name}>=0.6"] = float(correct[hi].mean()) if hi.any() else np.nan
        rows[f"accuracy_{name}<0.6"] = float(correct[lo].mean()) if lo.any() else np.nan
        rows[f"share_{name}>=0.6"] = float(hi.mean())
    rows["accuracy_overall"] = float(correct.mean())
    # technical confounder: is low confidence just low sequencing depth?
    lib = np.log1p(ctx.counts.sum(1))
    if confidence is not None:
        rows["confidence_spearman_vs_log_library"] = float(spearmanr(confidence, lib)[0])
    df = {"x": ctx.coords[:, 0], "y": ctx.coords[:, 1], "truth": truth, "label": labels, "correct": correct,
          "boundary_distance": bd, "confidence": confidence, "replicate_stability": stability, "replicate_agreement": agreement}
    return rows, df


# Expected layer of well-known DLPFC markers (Maynard et al. 2021 and cortical
# biology). Used only as a qualitative sanity check of modal-domain assignment.
EXPECTED_LAYER = {"AQP4": "L1", "GFAP": "L1", "HPCAL1": "L2", "CUX2": "L2/3", "FREM3": "L3", "RORB": "L4",
                  "PCP4": "L5", "TRABD2A": "L5", "KRT17": "L6", "MOBP": "WM", "MBP": "WM", "PLP1": "WM"}


def _layer_key(name):
    t = "".join(ch for ch in str(name).lower() if ch.isalnum())
    if "wm" in t or "white" in t:
        return "WM"
    digits = "".join(ch for ch in t if ch.isdigit())
    return f"L{digits}" if digits else str(name)


def marker_check(genes, pi, selected, modal_label_names):
    rows = []
    up = [str(g).upper() for g in genes]
    for g, exp in EXPECTED_LAYER.items():
        if g in up:
            i = up.index(g)
            got = _layer_key(modal_label_names[i]) if modal_label_names[i] is not None else None
            rows.append({"gene": g, "expected_layer": exp, "pi": float(pi[i]), "selected": bool(selected[i]),
                         "modal_layer": got, "matches": bool(selected[i]) and got is not None and got in exp.replace("L2/3", "L2 L3")})
    return rows


# Layer / white-matter markers reported for human DLPFC (Maynard et al. 2021; spatialLIBD).
DLPFC_MARKERS = ["MBP", "MOBP", "PLP1", "GFAP", "AQP4", "PCP4", "SNAP25", "HPCAL1", "CARTPT", "TRABD2A",
                 "KRT17", "CCK", "ENC1", "NEFL", "NEFH", "RORB", "CUX2", "HBB", "FABP7"]


def r6_global(ctx, seed=0):
    T = M.global_svg_tests(ctx.counts, ctx.coords)
    R = A.global_reference(T)
    C = A.castl_like(T, ctx.counts, ctx.coords, M.global_svg_tests, rng=np.random.default_rng(seed))
    lists = {"global intersection_rescued": R["intersection_rescued"], "global acat": R["acat"],
             "Castl-like Cauchy+BY": C["pval_agg"], "Castl-like NC-FDP": C["fdp_agg"], "Castl-like rank": C["rank_agg"]}
    rows = [{"approach": n, "n_called": int(v.sum())} for n, v in lists.items()]
    # pairwise overlap with the intersection reference
    for r in rows:
        r["overlap_with_intersection"] = float((lists[r["approach"]] & R["intersection_rescued"]).sum() / max(lists[r["approach"]].sum(), 1))
    genes = np.asarray(ctx.genes) if getattr(ctx, "genes", None) is not None else None
    if genes is not None:
        mk = np.isin(np.char.upper(genes.astype(str)), DLPFC_MARKERS)
        for r in rows:
            r["known_markers_present"] = int(mk.sum())
            r["known_markers_recovered"] = int((lists[r["approach"]] & mk).sum())
    return rows, lists, T
