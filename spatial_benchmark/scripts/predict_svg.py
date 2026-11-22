"""Rank overall SVGs in measured sections with fixed, training-free settings."""

import argparse
from hashlib import file_digest
import json
from importlib.metadata import version
from itertools import combinations
from pathlib import Path
import sys

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse, stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stbench.methods import gearys_c, knn_weights, morans_i, sparkx_like


CALLERS = ["moran_i", "negative_geary_c", "kernel_proxy"]
SEED = 590
N_REPLICATES = 20


def load_section(entry, data_dir):
    path = data_dir / entry["file"]
    with path.open("rb") as source:
        checksum = file_digest(source, "sha256").hexdigest()
    data = ad.read_h5ad(path)
    for column, value in entry.get("subset", {}).items():
        values = data.obs[column]
        if isinstance(value, (int, float)):
            matches = np.isclose(values.to_numpy(), value, rtol=0, atol=1e-8)
        else:
            matches = values == value
        data = data[matches]
    if data.n_obs < 10 or (not data.var_names.is_unique and "gene_ids" not in data.var):
        raise ValueError(f"Insufficient locations or ambiguous gene IDs in {path}")
    matrix = sparse.csr_matrix(data.X)
    if not np.isfinite(matrix.data).all() or (matrix.data < 0).any():
        raise ValueError(f"Invalid expression values in {path}")
    depth = np.asarray(matrix.sum(axis=1)).ravel()
    keep = depth > 0
    matrix = matrix[keep]
    coordinates = np.asarray(data.obsm["spatial"])[keep, :2]
    if len(coordinates) < 10 or not np.isfinite(coordinates).all() or (coordinates.std(0) == 0).any():
        raise ValueError(f"Missing or degenerate coordinates in {path}")
    ids = np.asarray(data.var.get("gene_ids", data.var_names), dtype=str)
    if len(set(ids)) != len(ids):
        raise ValueError(f"Gene IDs must be unique in {path}")
    detection = np.asarray((matrix > 0).mean(axis=0)).ravel()
    eligible = np.flatnonzero(detection >= 0.05)
    # The same screen is applied once, before spatial scoring or perturbation.
    eligible = eligible[np.lexsort((ids[eligible], -detection[eligible]))][:2000]
    if not len(eligible):
        raise ValueError(f"No eligible genes in {path}")
    x = matrix[:, eligible].toarray().astype(float)
    if entry["expression"] == "counts":
        if not np.equal(matrix.data, np.round(matrix.data)).all():
            raise ValueError(f"Expected integer counts in {path}")
        x = np.log1p(x * (1e4 / depth[keep])[:, None])
    elif entry["expression"] == "normalized":
        x = np.log1p(x)
    elif entry["expression"] != "log_normalized":
        raise ValueError(f"Unknown expression representation in {path}")
    available_locations = len(x)
    if entry.get("max_locations", available_locations) < available_locations:
        subset = np.sort(np.random.default_rng(SEED).choice(
            available_locations, entry["max_locations"], replace=False
        ))
        x, coordinates = x[subset], coordinates[subset]
    metadata = {**entry, "sha256": checksum, "locations": len(x), "available_locations": available_locations,
                "tested_genes": len(eligible)}
    genes = pd.DataFrame({"gene_id": ids[eligible], "gene": data.var_names[eligible],
                          "detection_fraction": detection[eligible]})
    return x, coordinates, genes, metadata


def score_section(x, coordinates):
    weights = knn_weights(coordinates, 6)
    moran, _ = morans_i(x, weights)
    # Geary's C materializes an edge-by-gene array. Bound its memory by batching.
    geary = np.concatenate([gearys_c(x[:, start:start + 64], weights)[0]
                            for start in range(0, x.shape[1], 64)])
    kernel = -np.log10(sparkx_like(x, coordinates))
    scores = np.column_stack([moran, -geary, kernel])
    scores[np.ptp(x, axis=0) == 0] = -np.inf
    if np.isnan(scores).any() or np.isposinf(scores).any():
        raise ValueError("Spatial scoring produced invalid values")
    ranked = stats.rankdata(scores, axis=0) / len(scores)
    graph = ranked[:, :2].mean(axis=1)
    kernel = ranked[:, 2]
    consensus = 0.75 * (graph + kernel) / 2 + 0.25 * np.maximum(graph, kernel)
    return scores, consensus


def ordered_ranks(score, gene_ids):
    order = np.lexsort((gene_ids, -score))
    ranks = np.empty(len(order), dtype=int)
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks


def predict(entry, data_dir, out):
    x, coordinates, genes, metadata = load_section(entry, data_dir)
    gene_ids = genes["gene_id"].to_numpy()
    top_k = min(len(genes), entry["top_k"])
    scores, consensus = score_section(x, coordinates)
    genes[CALLERS] = np.where(np.isfinite(scores), scores, np.nan)
    genes["ensemble_score"] = consensus
    genes["rank"] = ordered_ranks(consensus, gene_ids)
    rng = np.random.default_rng(SEED)
    ranks = []
    for _ in range(N_REPLICATES):
        subset = np.sort(rng.choice(len(x), int(0.8 * len(x)), replace=False))
        _, perturbed = score_section(x[subset], coordinates[subset])
        ranks.append(ordered_ranks(perturbed, gene_ids))
    ranks = np.asarray(ranks)
    genes["rank_p10"], genes["rank_median"], genes["rank_p90"] = np.quantile(ranks, [0.1, 0.5, 0.9], axis=0)
    genes["top_k_frequency"] = (ranks <= top_k).mean(axis=0)
    genes["top_k"] = genes["rank"] <= top_k
    genes = genes.sort_values("rank")
    genes.to_csv(out / f"{entry['id']}.csv", index=False, float_format="%.6g")
    print(f"{entry['id']}: {len(x)} locations, {len(genes)} genes", flush=True)
    return genes, metadata


def compare_sections(outputs):
    rows = []
    for (left, (a, ma)), (right, (b, mb)) in combinations(outputs.items(), 2):
        if ma["dataset"] != mb["dataset"]:
            continue
        shared = a.merge(b, on="gene_id", suffixes=("_a", "_b"))
        if len(shared) < 2:
            continue
        k = min(ma["top_k"], mb["top_k"], len(shared))
        top_a = set(shared.nsmallest(k, "rank_a")["gene_id"])
        top_b = set(shared.nsmallest(k, "rank_b")["gene_id"])
        rows.append({"section_a": left, "section_b": right, "shared_genes": len(shared),
                     "matched_k": k, "top_k_overlap": len(top_a & top_b),
                     "jaccard": len(top_a & top_b) / len(top_a | top_b),
                     "spearman": stats.spearmanr(shared["rank_a"], shared["rank_b"]).statistic})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    entries = json.loads(args.config.read_text())
    outputs = {entry["id"]: predict(entry, args.data_dir, args.out) for entry in entries}
    comparisons = compare_sections(outputs)
    if comparisons:
        pd.DataFrame(comparisons).to_csv(args.out / "section_overlap.csv", index=False, float_format="%.6g")
    manifest = {"samples": [metadata for _, metadata in outputs.values()], "seed": SEED,
                "spot_subsamples": N_REPLICATES, "location_fraction": 0.8,
                "versions": {package: version(package) for package in ("anndata", "numpy", "pandas", "scipy", "scikit-learn")}}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
