"""Spatial transcriptomics simulator with known domains and known SVGs.

Layout: a Visium-like hexagonal spot grid. Domains are curved cortical-style
layers (like DLPFC) or blobs. Counts are negative binomial with spot-level
library-size variation, so sparsity and mean-variance behaviour look like
NGS-based platforms.

Gene classes (ground truth):
  domain   - up-regulated in exactly one domain by a log fold-change (logFC)
  gradient - smooth global trend across the tissue (a global SVG, not
             domain-specific)
  null     - no spatial structure. A subset of null genes follows random,
             non-spatial "cell states" (spot-level heterogeneity that is real
             biology-like variation but not spatial); these mislead HVG
             ranking and expression-only clustering, as in real tissue.

Realism knobs: boundary softness (spots near a boundary express a mixture of
neighbouring domains), dispersion, marker strength and count depth.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SimData:
    counts: np.ndarray            # spots x genes (int)
    coords: np.ndarray            # spots x 2 (spot-spacing units)
    domains: np.ndarray           # spots (int labels 0..K-1)
    gene_class: np.ndarray        # genes: "domain" | "gradient" | "null"
    gene_domain: np.ndarray       # genes: target domain for "domain" genes, -1 otherwise
    gene_logfc: np.ndarray        # genes: logFC for domain genes, 0 otherwise
    meta: dict = field(default_factory=dict)

    @property
    def is_svg(self):
        return self.gene_class != "null"

    def subset_spots(self, idx):
        return SimData(self.counts[idx], self.coords[idx], self.domains[idx], self.gene_class,
                       self.gene_domain, self.gene_logfc, dict(self.meta))


def hex_grid(nx, ny):
    xs, ys = np.meshgrid(np.arange(nx, dtype=float), np.arange(ny, dtype=float) * np.sqrt(3) / 2)
    xs[1::2] += 0.5
    return np.c_[xs.ravel(), ys.ravel()]


def layered_domains(coords, k, rng, curvature=0.15):
    """Curved bands (cortex-like layers) with uneven widths."""
    x = (coords[:, 0] - coords[:, 0].min()) / np.ptp(coords[:, 0])
    y = (coords[:, 1] - coords[:, 1].min()) / np.ptp(coords[:, 1])
    depth = y + curvature * np.sin(np.pi * x + rng.uniform(0, 0.5))
    widths = rng.dirichlet(np.full(k, 4.0))
    cuts = np.quantile(depth, np.cumsum(widths)[:-1])
    return np.digitize(depth, cuts)


def blob_domains(coords, k, rng):
    """Voronoi-like blobs from random seed points."""
    centers = coords[rng.choice(len(coords), k, replace=False)]
    d = ((coords[:, None, :] - centers[None]) ** 2).sum(-1)
    return d.argmin(1)


def simulate(n_x=50, n_y=46, n_domains=6, n_genes=1000, n_domain_genes=150, n_gradient_genes=50,
             logfcs=(0.4, 0.8, 1.2, 1.6), layout="layers", base_mean=(0.2, 4.0), dispersion=0.6,
             lib_sd=0.4, boundary=1.5, n_states=4, n_state_genes=100, state_lfc=0.5, seed=0, noise_seed=None):
    """Defaults give a 'moderate' regime calibrated to Kang et al. (2025): domain
    ARI 0.06-0.6 across methods, ~50 % zeros, and an intersection-consensus SVG
    reference that recovers only ~60-70 % of true SVGs."""
    rng = np.random.default_rng(seed)
    coords = hex_grid(n_x, n_y)
    dom = layered_domains(coords, n_domains, rng) if layout == "layers" else blob_domains(coords, n_domains, rng)
    n = len(coords)
    # gene baselines: log-uniform means as in real data (many low-expressed genes)
    mu0 = np.exp(rng.uniform(np.log(base_mean[0]), np.log(base_mean[1]), n_genes))
    cls = np.array(["null"] * n_genes, dtype=object)
    target = np.full(n_genes, -1); lfc = np.zeros(n_genes)
    order = rng.permutation(n_genes)
    dg, gg = order[:n_domain_genes], order[n_domain_genes:n_domain_genes + n_gradient_genes]
    cls[dg] = "domain"; cls[gg] = "gradient"
    target[dg] = rng.integers(0, n_domains, len(dg))
    lfc[dg] = rng.choice(logfcs, len(dg))
    log_mu = np.tile(np.log(mu0), (n, 1))
    # soft domain membership: Gaussian-smoothed indicator over `boundary` spot spacings
    from sklearn.neighbors import NearestNeighbors
    member = np.eye(n_domains)[dom]
    if boundary > 0:
        nn = NearestNeighbors(radius=3 * boundary).fit(coords)
        dist, ind = nn.radius_neighbors(coords)
        member = np.stack([(np.exp(-d ** 2 / (2 * boundary ** 2))[:, None] * np.eye(n_domains)[dom[i]]).sum(0)
                           / np.exp(-d ** 2 / (2 * boundary ** 2)).sum() for d, i in zip(dist, ind)])
    for g in dg:  # natural-log fold change, weighted by membership of the target domain
        log_mu[:, g] += lfc[g] * member[:, target[g]]
    # non-spatial cell states (null genes only)
    nulls = order[n_domain_genes + n_gradient_genes:]
    state = rng.integers(0, n_states, n); state_orig = state
    sg = nulls[:n_state_genes]
    eff = rng.normal(0, state_lfc, (n_states, len(sg)))
    log_mu[:, sg] += eff[state]
    xy = (coords - coords.mean(0)) / coords.std(0)
    for g in gg:  # smooth global gradient along a random direction, amplitude ~ 1 on log scale
        theta = rng.uniform(0, 2 * np.pi)
        proj = xy @ np.array([np.cos(theta), np.sin(theta)])
        log_mu[:, g] += rng.uniform(0.4, 0.8) * proj
    # noise_seed gives an independent "adjacent section": same tissue, layout and
    # gene effects, new cell states, library sizes and counts.
    if noise_seed is not None:
        rng = np.random.default_rng(noise_seed)
        state = rng.integers(0, n_states, n)
        log_mu[:, sg] -= eff[state_orig]
        log_mu[:, sg] += eff[state]
    lib = np.exp(rng.normal(0, lib_sd, n))
    mu = np.exp(log_mu) * lib[:, None]
    r = 1.0 / dispersion
    counts = rng.negative_binomial(r, r / (r + mu))
    return SimData(counts.astype(np.int32), coords, dom, cls, target, lfc,
                   {"layout": layout, "n_domains": n_domains, "seed": seed, "dispersion": dispersion,
                    "boundary": boundary, "state_genes": sg.tolist(), "membership_max": member.max(1)})


# ----------------------------------------------------------------- perturbations
def thin_counts(counts, frac, rng):
    """Binomial thinning: simulates lower sequencing depth."""
    return counts if frac >= 1 else rng.binomial(counts, frac).astype(np.int32)


def jitter(coords, sd, rng):
    """Coordinate noise (registration / spot-position error), in spot spacings."""
    return coords if sd <= 0 else coords + rng.normal(0, sd, coords.shape)


def perturb(sim, rng, depth=1.0, spot_frac=1.0, coord_sd=0.0):
    """Return a perturbed copy and the indices of retained spots."""
    idx = np.sort(rng.choice(len(sim.coords), int(round(spot_frac * len(sim.coords))), replace=False)) if spot_frac < 1 else np.arange(len(sim.coords))
    out = sim.subset_spots(idx)
    out.counts = thin_counts(out.counts, depth, rng)
    out.coords = jitter(out.coords, coord_sd, rng)
    return out, idx
