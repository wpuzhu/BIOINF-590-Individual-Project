"""Compare saved SVG ranks against simpler scores without resampling expression."""

import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


CALLERS = ["moran_i", "negative_geary_c", "kernel_proxy"]


def load_rankings(run_dir):
    for sample in json.loads((run_dir / "manifest.json").read_text())["samples"]:
        table = pd.read_csv(run_dir / f"{sample['id']}.csv").set_index("gene_id")
        percentiles = table[CALLERS].fillna(-np.inf).rank(pct=True)
        graph = percentiles[CALLERS[:2]].mean(axis=1)
        scores = {
            **{caller: percentiles[caller] for caller in CALLERS},
            "flat_caller_mean": percentiles.mean(axis=1),
            "equal_family_mean": (graph + percentiles["kernel_proxy"]) / 2,
            "max_family": np.maximum(graph, percentiles["kernel_proxy"]),
            "ensemble_75_25": -table["rank"],
        }
        ranks = {}
        for method, score in scores.items():
            ranked = score.rename("score").reset_index().sort_values(
                ["score", "gene_id"], ascending=[False, True]
            )
            ranks[method] = pd.Series(range(1, len(table) + 1), index=ranked["gene_id"])
        yield sample, table, pd.DataFrame(ranks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    samples = [item for run in args.runs for item in load_rankings(run)]
    sample_rows = []
    for sample, table, ranks in samples:
        for method in ranks:
            top = ranks[method].nsmallest(sample["top_k"]).index
            sample_rows.append({
                "section": sample["id"], "dataset": sample["dataset"], "method": method,
                "top_k": len(top), "mitochondrial_symbols": int(table.loc[top, "gene"].str.upper().str.startswith("MT-").sum()),
                "score_distinct_kernel": table["kernel_proxy"].nunique(),
                "ensemble_stable_at_0_8": int((table.loc[top, "top_k_frequency"] >= 0.8).sum()),
            })
    pair_rows = []
    for (a, _, ranks_a), (b, _, ranks_b) in combinations(samples, 2):
        if a["dataset"] != b["dataset"]:
            continue
        shared = ranks_a.index.intersection(ranks_b.index)
        k = min(a["top_k"], b["top_k"], len(shared))
        if k < 2:
            continue
        for method in ranks_a:
            left, right = ranks_a.loc[shared, method], ranks_b.loc[shared, method]
            overlap = len(set(left.nsmallest(k).index) & set(right.nsmallest(k).index))
            pair_rows.append({
                "section_a": a["id"], "section_b": b["id"], "method": method,
                "shared_genes": len(shared), "matched_k": k, "overlap": overlap,
                "jaccard": overlap / (2 * k - overlap), "spearman": spearmanr(left, right).statistic,
            })
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sample_rows).to_csv(args.out / "score_diagnostics.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(args.out / "rank_ablation.csv", index=False, float_format="%.6g")


if __name__ == "__main__":
    main()
