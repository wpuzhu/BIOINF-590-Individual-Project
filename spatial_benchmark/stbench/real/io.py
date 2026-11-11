"""Loading real spatial datasets into the AnnData-free Context used everywhere else.

Primary dataset: human DLPFC, 10x Visium, 12 sections from 3 donors, with manual
layer annotation (Maynard et al., Nat Neurosci 2021). It is used by Kang et al.
2025, EnSDD, STCC, SpatialESD, InSTability and most domain-detection papers.

Sources tried in order (the first that works is used):
  1. local:  a .h5ad you already have (`path=`)
  2. hf:     Hugging Face dataset `han-shu/st_datasets`, files DLPFC/DLPFC_<id>.h5ad
  3. sodb:   pysodb  `SODB().load_experiment('maynard2021trans', '<id>')`

Adjacent-section pairs (10 um apart, same donor) give a truth-free replication
test: 151507/151508, 151509/151510, 151669/151670, 151671/151672,
151673/151674, 151675/151676.
"""
import os
import warnings

import numpy as np

from ..workflow import Context

DLPFC_SAMPLES = ["151507", "151508", "151509", "151510", "151669", "151670",
                 "151671", "151672", "151673", "151674", "151675", "151676"]
ADJACENT_PAIRS = [("151507", "151508"), ("151509", "151510"), ("151669", "151670"),
                  ("151671", "151672"), ("151673", "151674"), ("151675", "151676")]
LABEL_CANDIDATES = ["layer_guess_reordered", "layer_guess", "Region", "ground_truth", "layer", "Layer",
                    "spatialLIBD", "annotation", "Annotation", "label", "domain", "celltype", "cell_type", "cluster"]


def _read_h5ad(path):
    import anndata as ad
    return ad.read_h5ad(path)


def load_dlpfc(sample, sources=("local", "hf", "sodb"), cache_dir="data", path=None):
    errors = []
    for src in sources:
        try:
            if src == "local":
                p = path or os.path.join(cache_dir, f"DLPFC_{sample}.h5ad")
                if os.path.exists(p):
                    return _read_h5ad(p), f"local:{p}"
                raise FileNotFoundError(p)
            if src == "hf":
                from huggingface_hub import hf_hub_download
                p = hf_hub_download("han-shu/st_datasets", f"DLPFC/DLPFC_{sample}.h5ad", repo_type="dataset",
                                    cache_dir=cache_dir)
                return _read_h5ad(p), "hf:han-shu/st_datasets"
            if src == "sodb":
                import pysodb
                return pysodb.SODB().load_experiment("maynard2021trans", sample), "sodb:maynard2021trans"
        except Exception as e:  # try the next source
            errors.append(f"{src}: {type(e).__name__}: {e}")
    raise RuntimeError(f"Could not load DLPFC {sample}. Tried: " + " | ".join(errors))


def sim_context(sample, max_genes=6000):
    """Offline test source: 'simS' and 'simSb' are two independent sections of the
    same simulated tissue (seed S), mimicking an adjacent-section pair."""
    from ..simulate import simulate
    from ..methods import hvg, lognorm
    seed = int(sample[3:].rstrip("b"))
    sim = simulate(seed=seed, noise_seed=(10_000 + seed) if sample.endswith("b") else None)
    ctx = Context(sim.counts, sim.coords, [f"g{i}" for i in range(sim.counts.shape[1])])
    ctx.cluster_genes = np.sort(hvg(lognorm(sim.counts), 500))
    names = [f"D{d}" for d in range(sim.domains.max() + 1)]
    ctx.extra.update({"count_source": "simulated", "label_col": "truth", "n_spots": len(sim.coords), "n_genes": sim.counts.shape[1],
                      "label_names": names, "labels": sim.domains, "is_svg": sim.is_svg})
    return ctx


def load_squidpy(name):
    """Optional second-platform datasets bundled by squidpy (labels are cell
    classes or author clusters, not curated domains)."""
    import squidpy as sq
    loaders = {"seqfish": (sq.datasets.seqfish, "celltype_mapped_refined"),
               "merfish": (sq.datasets.merfish, "Cell_class"),
               "slideseqv2": (sq.datasets.slideseqv2, "cluster"),
               "visium_hne": (sq.datasets.visium_hne_adata, "cluster")}
    f, label = loaders[name]
    adata = f()
    return adata, f"squidpy:{name}", label


def find_label(adata, preferred=None):
    """Known column names first; otherwise any categorical column whose values
    look like cortical layers (contain 'WM' or 'Layer'/'L1'..'L6')."""
    cols = ([preferred] if preferred else []) + LABEL_CANDIDATES
    for c in cols:
        if c and c in adata.obs.columns and adata.obs[c].nunique() > 1:
            return c
    for c in adata.obs.columns:
        vals = set(map(str, adata.obs[c].dropna().unique()))
        if 3 <= len(vals) <= 12 and ("WM" in vals or any(v.startswith(("Layer", "L1", "L2", "L3")) for v in vals)):
            return c
    return None


def get_counts(adata):
    """Raw integer counts from layers['counts'], .raw, or .X (in that order)."""
    import scipy.sparse as sp
    cands = []
    if "counts" in getattr(adata, "layers", {}):
        cands.append(("layers['counts']", adata.layers["counts"], adata.var_names))
    if adata.raw is not None:
        cands.append(("raw.X", adata.raw.X, adata.raw.var_names))
    cands.append(("X", adata.X, adata.var_names))
    for name, M, var in cands:
        sample = M[:200].toarray() if sp.issparse(M) else np.asarray(M[:200])
        if np.allclose(sample, np.round(sample)) and sample.min() >= 0:
            return M, list(map(str, var)), name
    warnings.warn("No integer count matrix found; count splitting assumes counts. Using X rounded.")
    M = adata.X
    return M, list(map(str, adata.var_names)), "X (non-integer, rounded)"


def spacing_units(coords):
    """Rescale coordinates so the median nearest-neighbour distance is 1."""
    from sklearn.neighbors import NearestNeighbors
    d, _ = NearestNeighbors(n_neighbors=2).fit(coords).kneighbors(coords)
    return (coords - coords.min(0)) / np.median(d[:, 1])


def to_context(adata, label_col=None, min_spot_frac=0.05, max_genes=6000, n_cluster_genes=3000, drop_unlabelled=True):
    """Build a Context: integer counts (dense), coordinates in spot-spacing units,
    gene filter = detected in >= min_spot_frac of spots (top max_genes by mean),
    cluster genes = highly variable genes among those."""
    import scipy.sparse as sp
    label_col = find_label(adata, label_col)
    keep = np.ones(adata.n_obs, bool)
    labels = None
    if label_col is not None:
        lab = adata.obs[label_col].astype(str).values
        if drop_unlabelled:
            keep &= ~np.isin(lab, ["nan", "NA", "None", ""])
        labels = lab
    M, genes, src = get_counts(adata)
    M = M[keep]
    M = M.tocsc() if sp.issparse(M) else np.asarray(M)
    det = np.asarray((M > 0).sum(0)).ravel() / M.shape[0]
    mean = np.asarray(M.mean(0)).ravel()
    cand = np.where(det >= min_spot_frac)[0]
    cand = cand[np.argsort(-mean[cand])[:max_genes]]
    cand = np.sort(cand)
    X = M[:, cand]
    X = (X.toarray() if sp.issparse(X) else np.asarray(X)).astype(np.int32)
    coords = np.asarray(adata.obsm["spatial"])[keep].astype(float)
    ctx = Context(X, spacing_units(coords), [genes[i] for i in cand])
    from ..methods import hvg, lognorm
    ctx.cluster_genes = np.sort(hvg(lognorm(X), min(n_cluster_genes, X.shape[1])))
    meta = {"count_source": src, "label_col": label_col, "n_spots": int(keep.sum()), "n_genes": int(X.shape[1]),
            "obs_names": list(map(str, np.asarray(adata.obs_names)[keep]))}
    if labels is not None:
        lab = labels[keep]
        uniq = sorted(set(lab))
        meta["label_names"] = uniq
        meta["labels"] = np.array([uniq.index(v) for v in lab])
    ctx.extra.update(meta)
    return ctx
