"""Shared workflow for simulations and real data.

A *domain method* is any callable  f(ctx, k, seed) -> labels  where ctx is a
Context (counts, coords, cached embedding). CPU methods from methods.py are
wrapped automatically. GPU methods (stbench.real.gpu_methods) use the same
signature, so the ensemble, pruning and SplitStab code never needs to know
which kind it is calling.
"""
from dataclasses import dataclass, field

import numpy as np

from . import ensemble as E
from . import methods as M
from .domain_svg import add_negative_controls, mb_bound, nc_calibrated_threshold, spatial_gate, test_domains
from .splitting import count_split


@dataclass
class Context:
    counts: np.ndarray                  # spots x genes (dense int)
    coords: np.ndarray                  # spots x 2
    genes: list = None
    cluster_genes: np.ndarray = None    # column indices used for embedding (None = all)
    extra: dict = field(default_factory=dict)  # e.g. AnnData for GPU wrappers, spot ids
    _pcs: dict = field(default_factory=dict)

    def pcs(self, seed=0, n_pcs=20, n_hvg=2000):
        if seed not in self._pcs:
            X = self.counts if self.cluster_genes is None else self.counts[:, self.cluster_genes]
            self._pcs[seed] = M.embed(X, n_pcs=n_pcs, n_hvg=min(n_hvg, X.shape[1]), seed=seed)
        return self._pcs[seed]

    def subset(self, idx, counts=None):
        return Context(self.counts[idx] if counts is None else counts, self.coords[idx], self.genes,
                       self.cluster_genes, {**self.extra, "spot_index": np.asarray(idx)})


def cpu_method(name):
    f = M.DOMAIN_METHODS[name]
    def run(ctx, k, seed):
        return f(ctx.pcs(seed), ctx.coords, k, seed)
    run.__name__ = name
    return run


CPU_METHODS = {name: cpu_method(name) for name in M.DOMAIN_METHODS}


def run_members(ctx, k, methods, n_reps=3, rng=None, mild=True, log=None):
    """Each method runs n_reps times on the same spots: new seed and, for r > 0
    and mild=True, light depth thinning (0.8) + 0.3-spacing coordinate jitter."""
    rng = rng or np.random.default_rng(0)
    members = {m: [] for m in methods}
    for r in range(n_reps):
        c = ctx
        if mild and r > 0:
            c = Context(rng.binomial(ctx.counts, 0.8).astype(ctx.counts.dtype), ctx.coords + rng.normal(0, 0.3, ctx.coords.shape),
                        ctx.genes, ctx.cluster_genes, ctx.extra)
        for m, f in methods.items():
            try:
                members[m].append(np.asarray(f(c, k, r)).astype(int))
            except Exception as e:  # a failing method is recorded, not fatal
                if log is not None:
                    log.append(f"{m} rep {r} failed: {type(e).__name__}: {e}")
    return {m: v for m, v in members.items() if len(v)}


# Per-replicate test rules. "strict" was the v2 default; "liberal" carries over the
# permissive per-run rule of v1 (which had high recall) and relies on the spatial
# gate + stability + negative controls for error control. "liberal_nocore" is an
# ablation that ignores domain confidence.
RULES = {
    "strict": dict(q_thr=0.05, fold=1.5),
    "liberal": dict(q_thr=0.10, fold=1.2),
    "liberal_nocore": dict(q_thr=0.10, fold=1.2, core=0.0),
}
PRIMARY_RULE = "liberal"


def splitstab(ctx, k, methods, B=20, spot_frac=0.8, eps=0.5, nc_frac=0.1, n_reps=2, alpha=0.1,
              test_kw=None, rng=None, aggregation="pce", log=None, progress=None, gate=True,
              rules=None, primary=None, reference=None, record_members=False):
    """Combined framework, gene level (SplitStab).

    Each of B replicates: subsample spots -> count split (X1, X2) -> domain members
    on X1 -> PCE consensus with per-spot confidence -> for every rule, core-spot NB
    test on X2 -> spatial gate on X2 (ACAT of Moran's I + SPARK-X-like). Synthetic
    NB negative controls ride along. All rules share the same domains, so the
    ablation is paired.

    The spatial gate is also the *global* SVG call of that replicate: one shared
    spatial core produces both the domain-SVG gate and a stability-selected global
    SVG list (v1's ACAT reference, now on held-out counts and with a pi).

    Replicate labels are aligned to `reference` (full-data PCE labels, computed
    here if not given), so modal domains are comparable across replicates and each
    spot gets a replicate-agreement score (how often it keeps its domain when the
    tissue is resampled).

    Returns the primary rule's results at top level (backward compatible) and all
    rules under "by_rule" / "by_rule_nogate"."""
    rng = rng or np.random.default_rng(0)
    if rules is None:
        rules = {"main": test_kw or {}} if test_kw is not None else dict(RULES)
    primary = primary or (PRIMARY_RULE if PRIMARY_RULE in rules else next(iter(rules)))
    n = len(ctx.coords); G = ctx.counts.shape[1]
    if reference is None:
        m0 = run_members(ctx, k, methods, n_reps, rng, mild=False, log=log)
        reference = E.pce(m0, ctx.coords, k, seed=0)["labels"]
    reference = np.asarray(reference)
    aug, n_nc = add_negative_controls(ctx.counts, nc_frac, rng)
    calls = {r: np.zeros((B, G + n_nc), bool) for r in rules}
    calls_ng = {r: np.zeros((B, G + n_nc), bool) for r in rules}
    gate_calls = np.zeros((B, G + n_nc), bool)
    dom_votes = {r: np.zeros((G, k + 1), int) for r in rules}
    n_called = {r: [] for r in rules}
    agree = np.zeros(n); seen = np.zeros(n); conf_sum = np.zeros(n); comem = np.zeros(n)
    kept_log, rec = [], []
    for b in range(B):
        idx = np.sort(rng.choice(n, int(round(spot_frac * n)), replace=False))
        X1, X2 = count_split(aug[idx], eps, rng)
        sub = Context(X1[:, :G], ctx.coords[idx], ctx.genes, ctx.cluster_genes, {**ctx.extra, "spot_index": idx})
        members = run_members(sub, k, methods, n_reps, rng, mild=False, log=log)
        if not members:
            continue
        res = E.pce(members, sub.coords, k, seed=b) if aggregation == "pce" else E.ensemble(members, sub.coords, k, aggregation, seed=b)
        lab = E.align(reference[idx], res["labels"], k)
        conf = res.get("confidence")
        kept_log.append(res.get("kept"))
        agree[idx] += (lab == reference[idx]); seen[idx] += 1
        # alignment-free co-membership stability: share of the spot's reference-domain
        # mates (in this subsample) that the replicate puts in the same domain as the spot
        ref_s = reference[idx]; raw = np.asarray(res["labels"])
        tab = np.zeros((ref_s.max() + 1, raw.max() + 1)); np.add.at(tab, (ref_s, raw), 1)
        comem[idx] += tab[ref_s, raw] / np.maximum(tab.sum(1)[ref_s], 1)
        if conf is not None:
            conf_sum[idx] += conf
        if record_members:
            rec.append({"idx": idx, "members": {m: v[0] for m, v in members.items()}, "kept": res.get("kept")})
        g_ok = spatial_gate(X2, sub.coords)[0] if gate else np.ones(G + n_nc, bool)
        gate_calls[b] = g_ok
        for r, kw in rules.items():
            hit, Q, LFC = test_domains(X2, lab, conf, **kw)
            called = hit.any(1)
            calls_ng[r][b] = called
            calls[r][b] = called & g_ok
            n_called[r].append(int(calls[r][b, :G].sum()))
            Qm = np.where(hit, Q, np.inf)[:G]
            best = np.where(calls[r][b, :G], np.argmin(Qm, 1), k)
            np.add.at(dom_votes[r], (np.arange(G), best), 1)
        if progress is not None:
            progress(b + 1, B)

    def summarise(C, votes, nc_list):
        pi = C[:, :G].mean(0); pi_nc = C[:, G:].mean(0)
        thr, table = nc_calibrated_threshold(pi, pi_nc, alpha)
        for row in table:
            row["mb_bound"] = mb_bound(np.mean(nc_list) if nc_list else 0.0, G, row["pi_thr"])
        out = {"pi": pi, "pi_nc": pi_nc, "threshold": thr, "fdp_table": table, "selected": pi >= thr,
               "mean_calls_per_replicate": float(np.mean(nc_list)) if nc_list else 0.0}
        if votes is not None:
            out["modal_domain"] = votes[:, :k].argmax(1)
            v = votes[:, :k].sum(1)
            out["domain_purity"] = np.where(v > 0, votes[:, :k].max(1) / np.maximum(v, 1), np.nan)
        return out

    by_rule = {r: summarise(calls[r], dom_votes[r], n_called[r]) for r in rules}
    by_rule_ng = {r: summarise(calls_ng[r], None, [int(c[:G].sum()) for c in calls_ng[r]]) for r in rules}
    glob = summarise(gate_calls, None, [int(c[:G].sum()) for c in gate_calls])
    out = dict(by_rule[primary])
    out.update({"primary_rule": primary, "by_rule": by_rule, "by_rule_nogate": by_rule_ng, "global": glob,
                "reference_labels": reference,
                "spot_agreement": np.where(seen > 0, agree / np.maximum(seen, 1), np.nan),
                "spot_stability": np.where(seen > 0, comem / np.maximum(seen, 1), np.nan),
                "spot_mean_confidence": np.where(seen > 0, conf_sum / np.maximum(seen, 1), np.nan),
                "kept_methods": kept_log, "members_record": rec})
    return out


def single_split(ctx, k, methods, rng=None, eps=0.5, n_reps=2, rules=None):
    """Ablation: ONE count split on all spots (no stability), PCE on X1, rules on X2, gate."""
    rng = rng or np.random.default_rng(0)
    rules = rules or dict(RULES)
    X1, X2 = count_split(ctx.counts, eps, rng)
    sub = Context(X1, ctx.coords, ctx.genes, ctx.cluster_genes, ctx.extra)
    res = E.pce(run_members(sub, k, methods, n_reps, rng, mild=False), ctx.coords, k, seed=0)
    g = spatial_gate(X2, ctx.coords)[0]
    out = {}
    for r, kw in rules.items():
        hit, _, _ = test_domains(X2, res["labels"], res.get("confidence"), **kw)
        out[r] = hit.any(1) & g
        out[r + "_nogate"] = hit.any(1)
    return out


def v1_stability(ctx, k, methods, B=20, rng=None, pi_thr=0.6, n_reps=2):
    """v1 (pilot) approach: B perturbed replicates (depth 0.8, 80 % spots, jitter),
    PCE domains, liberal SpaGCN rule on the SAME counts, keep genes with pi >= pi_thr."""
    from .simulate import thin_counts
    rng = rng or np.random.default_rng(0)
    n = len(ctx.coords); G = ctx.counts.shape[1]
    C = np.zeros((B, G), bool)
    for b in range(B):
        idx = np.sort(rng.choice(n, int(round(0.8 * n)), replace=False))
        X = thin_counts(ctx.counts[idx], 0.8, rng)
        sub = Context(X, ctx.coords[idx] + rng.normal(0, 0.3, (len(idx), 2)), ctx.genes, ctx.cluster_genes, ctx.extra)
        res = E.pce(run_members(sub, k, methods, n_reps, rng, mild=False), sub.coords, k, seed=b)
        hit, _ = M.domain_svgs(X, res["labels"], p_thr=0.05, in_frac=0.3, fold=1.2)
        C[b] = hit.any(1)
    pi = C.mean(0)
    return {"pi": pi, "selected": pi >= pi_thr, "mb_bound": mb_bound(C.sum(1).mean(), G, pi_thr)}


def paper_pipeline(ctx, k, methods, rng=None, strict=True):
    """Reference pipeline of Kang et al. (2025): one run per method -> aligned
    majority vote -> SpaGCN rule on the SAME counts (no split)."""
    rng = rng or np.random.default_rng(0)
    members = run_members(ctx, k, methods, 1, rng, mild=False)
    res = E.ensemble(members, ctx.coords, k, "majority")
    kw = dict(p_thr=0.05, in_frac=0.8, fold=1.5) if strict else dict(p_thr=0.05, in_frac=0.3, fold=1.2)
    hit, _ = M.domain_svgs(ctx.counts, res["labels"], **kw)
    return hit.any(1), res
