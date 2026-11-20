"""End-to-end pipeline pieces used by the pilot experiments."""
import numpy as np

from . import methods as M
from . import ensemble as E
from .simulate import perturb


def domain_members(sim, k, n_reps=5, rng=None, mild=True, methods=None):
    """Each method run n_reps times: different seed and, if `mild`, light depth
    thinning (0.8) + coordinate jitter (0.3 spacing) on the SAME spots, so that
    replicate labels are comparable spot-by-spot."""
    rng = rng or np.random.default_rng(0)
    methods = methods or list(M.DOMAIN_METHODS)
    members = {m: [] for m in methods}
    for r in range(n_reps):
        s = sim
        if mild and r > 0:
            s, _ = perturb(sim, rng, depth=0.8, coord_sd=0.3)
        pcs = M.embed(s.counts, seed=r)
        for m in methods:
            members[m].append(M.DOMAIN_METHODS[m](pcs, s.coords, k, r))
    return members


def run_once(sim, k, seed, n_reps=3, aggregation="cac_weighted_pruned"):
    """Domains (ensemble) -> domain-specific SVG calls, plus global tests."""
    rng = np.random.default_rng(seed)
    members = domain_members(sim, k, n_reps, rng)
    ens = E.ensemble(members, sim.coords, k, aggregation, seed=seed)
    hit, _ = M.domain_svgs(sim.counts, ens["labels"])
    tests = M.global_svg_tests(sim.counts, sim.coords)
    return members, ens, hit, tests
