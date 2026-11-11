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


def splitstab(ctx, k, methods, B=20, spot_frac=0.8, eps=0.5, nc_frac=0.1, n_reps=2, alpha=0.1,
              test_kw=None, rng=None, aggregation="pce", log=None, progress=None, gate=True):
    """SplitStab: B replicates of (subsample spots -> count split -> domains on X1
    -> pruned ensemble -> core-spot test on X2), with appended negative controls.
    Returns dict with pi (real genes), pi_nc, chosen threshold, FDP table,
    selected genes and their modal domain."""
    rng = rng or np.random.default_rng(0)
    test_kw = test_kw or {}
    G = ctx.counts.shape[1]
    aug, n_nc = add_negative_controls(ctx.counts, nc_frac, rng)
    calls = np.zeros((B, G + n_nc), bool)
    dom_votes = np.zeros((G, k + 1), int)
    n_called = []
    kept_log = []
    for b in range(B):
        idx = np.sort(rng.choice(len(ctx.coords), int(round(spot_frac * len(ctx.coords))), replace=False))
        X1, X2 = count_split(aug[idx], eps, rng)
        sub = Context(X1[:, :G], ctx.coords[idx], ctx.genes, ctx.cluster_genes, {**ctx.extra, "spot_index": idx})
        members = run_members(sub, k, methods, n_reps, rng, mild=False, log=log)
        if not members:
            continue
        if aggregation == "pce":
            res = E.pce(members, sub.coords, k, seed=b)
        else:
            res = E.ensemble(members, sub.coords, k, aggregation, seed=b)
        kept_log.append(res.get("kept"))
        hit, Q, LFC = test_domains(X2, res["labels"], res.get("confidence"), **test_kw)
        called = hit.any(1)
        if gate:
            called &= spatial_gate(X2, sub.coords)[0]
        calls[b] = called
        n_called.append(int(called[:G].sum()))
        best = np.where(called[:G], np.argmin(Q[:G], 1), k)
        np.add.at(dom_votes, (np.arange(G), best), 1)
        if progress is not None:
            progress(b + 1, B)
    pi = calls[:, :G].mean(0); pi_nc = calls[:, G:].mean(0)
    thr, table = nc_calibrated_threshold(pi, pi_nc, alpha)
    for row in table:
        row["mb_bound"] = mb_bound(np.mean(n_called), G, row["pi_thr"])
    selected = pi >= thr
    return {"pi": pi, "pi_nc": pi_nc, "threshold": thr, "fdp_table": table, "selected": selected,
            "modal_domain": dom_votes[:, :k].argmax(1), "mean_calls_per_replicate": float(np.mean(n_called)),
            "kept_methods": kept_log}


def paper_pipeline(ctx, k, methods, rng=None, strict=True):
    """Reference pipeline of Kang et al. (2025): one run per method -> aligned
    majority vote -> SpaGCN rule on the SAME counts (no split)."""
    rng = rng or np.random.default_rng(0)
    members = run_members(ctx, k, methods, 1, rng, mild=False)
    res = E.ensemble(members, ctx.coords, k, "majority")
    kw = dict(p_thr=0.05, in_frac=0.8, fold=1.5) if strict else dict(p_thr=0.05, in_frac=0.3, fold=1.2)
    hit, _ = M.domain_svgs(ctx.counts, res["labels"], **kw)
    return hit.any(1), res
