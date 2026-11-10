"""Pilot study for the proposal (simulated data with known truth).

E1  Aggregating domain results: single methods vs majority vote (paper) vs
    co-association consensus (unweighted / weighted / weighted + pruned);
    pruning curve (keep top-q by label-free score) vs the oracle order;
    calibration of per-spot confidence.
E2  Reference bias: domain-specific SVG calls from every partition evaluated
    against (a) the true SVGs and (b) the paper's intersection-consensus
    reference; does the reference change the method ranking?
E3  Global SVG reference: single tests vs intersection (+ Moran rescue) vs
    ACAT vs rank aggregation.
E4  Robustness: one-factor-at-a-time perturbations (sequencing depth, spot
    dropout, coordinate jitter) applied to the full pipeline.
E5  Uncertainty: stability selection of domain-specific SVGs over B
    perturbation replicates; calibration of selection probabilities and the
    Meinshausen-Buhlmann error bound; bootstrap probability that each domain
    method ranks first.

Usage: python scripts/run_pilot.py --out results/pilot --workers 2 [--quick]
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score as ari

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stbench import ensemble as E, methods as M, pipeline as P, svg_aggregate as A  # noqa: E402
from stbench.simulate import simulate, perturb  # noqa: E402

K = 6
DATASETS = [(layout, seed) for layout in ["layers", "blobs"] for seed in [1, 2, 3]]
AGGS = ["majority", "cac", "cac_weighted", "cac_weighted_pruned"]


def truth_sets(sim):
    return {"true_svg": sim.is_svg, "true_domain_svg": sim.gene_class == "domain"}


def e1_e2_e3(args):
    layout, seed = args
    sim = simulate(layout=layout, seed=seed)
    rng = np.random.default_rng(seed)
    members = P.domain_members(sim, K, 5, rng)
    tests = M.global_svg_tests(sim.counts, sim.coords)
    ref = A.global_reference(tests)
    truth = truth_sets(sim)
    rows, prune_rows, calib_rows, svg_rows, ref_rows = [], [], [], [], []
    partitions = {}
    for m, labs in members.items():
        for r, l in enumerate(labs):
            rows.append({"layout": layout, "seed": seed, "approach": m, "kind": "single", "rep": r, "ari": ari(sim.domains, l)})
        partitions[m] = labs[0]
    for mode in AGGS:
        e = E.ensemble(members, sim.coords, K, mode, seed=seed)
        rows.append({"layout": layout, "seed": seed, "approach": mode, "kind": "ensemble", "rep": 0,
                     "ari": ari(sim.domains, e["labels"]), "kept": ",".join(e["kept"])})
        partitions[mode] = e["labels"]
        # confidence calibration: align consensus to truth, accuracy per confidence bin
        aligned = E.align(sim.domains, e["labels"], K)
        correct = aligned == sim.domains
        for lo, hi in [(0, .5), (.5, .6), (.6, .7), (.7, .8), (.8, .9), (.9, 1.01)]:
            sel = (e["confidence"] >= lo) & (e["confidence"] < hi)
            if sel.sum():
                calib_rows.append({"layout": layout, "seed": seed, "mode": mode, "bin": f"{lo:.1f}-{min(hi,1):.1f}",
                                   "mid": (lo + min(hi, 1)) / 2, "n": int(sel.sum()), "accuracy": float(correct[sel].mean()),
                                   "mean_conf": float(e["confidence"][sel].mean())})
    # pruning curve: keep top-q by label-free score vs by oracle (true mean ARI)
    scores = E.method_scores(members, sim.coords)
    oracle = sorted(members, key=lambda m: -np.mean([ari(sim.domains, l) for l in members[m]]))
    label_free = sorted(members, key=lambda m: -scores[m]["score"])
    for order_name, order in [("label_free", label_free), ("oracle", oracle)]:
        for q in range(1, len(order) + 1):
            sub = {m: members[m] for m in order[:q]}
            e = E.ensemble(sub, sim.coords, K, "cac_weighted", seed=seed)
            prune_rows.append({"layout": layout, "seed": seed, "order": order_name, "q": q, "ari": ari(sim.domains, e["labels"]),
                               "methods": ",".join(order[:q])})
    # E2: domain-specific SVGs from every partition, two references
    consensus_ref = ref["intersection_rescued"]
    calls = {}
    for name, lab in partitions.items():
        hit, _ = M.domain_svgs(sim.counts, lab)
        called = hit.any(1); calls[name] = called
        for ref_name, t in [("truth (all SVGs)", truth["true_svg"]), ("truth (domain SVGs)", truth["true_domain_svg"]),
                            ("intersection consensus", consensus_ref)]:
            pr = A.precision_recall(called, t)
            svg_rows.append({"layout": layout, "seed": seed, "partition": name, "reference": ref_name,
                             "ari": ari(sim.domains, lab), **pr})
    single = [m for m in members]
    agree = A.agreement(np.array([calls[m] for m in single]))
    # E3: global reference quality
    for name in ["intersection", "intersection_rescued", "acat"]:
        ref_rows.append({"layout": layout, "seed": seed, "reference": name, **A.precision_recall(ref[name], truth["true_svg"])})
    for t, (score, p) in tests.items():
        rec = {"layout": layout, "seed": seed, "reference": t, "aupr": A.aupr(score, truth["true_svg"])}
        if p is not None:
            rec.update(A.precision_recall(M.bh(p) < 0.05, truth["true_svg"]))
            rec["null_fpr_at_0.05"] = float(((p < 0.05) & ~truth["true_svg"]).sum() / (~truth["true_svg"]).sum())
        ref_rows.append(rec)
    ref_rows.append({"layout": layout, "seed": seed, "reference": "rank_aggregation", "aupr": A.aupr(ref["rank_score"], truth["true_svg"])})
    ref_rows.append({"layout": layout, "seed": seed, "reference": "acat (score)", "aupr": A.aupr(-np.log10(ref["acat_p"] + 1e-300), truth["true_svg"])})
    return {"e1": rows, "prune": prune_rows, "calib": calib_rows, "e2": svg_rows, "e3": ref_rows,
            "agreement": {"layout": layout, "seed": seed, "agreement_single_methods": agree},
            "scores": {"layout": layout, "seed": seed, **{m: v for m, v in scores.items()}}}


# SpaGCN rule as used in the paper (strict) and a liberal setting (paper Fig. 5B trade-off)
RULES = {"strict": dict(p_thr=0.05, in_frac=0.8, fold=1.5), "liberal": dict(p_thr=0.05, in_frac=0.3, fold=1.2)}

AXES = {"depth": [1.0, 0.5, 0.25, 0.1], "spot_frac": [1.0, 0.8, 0.6, 0.4], "coord_sd": [0.0, 0.5, 1.0, 2.0]}


def e4(args):
    layout, seed, axis, level, rep = args
    base = simulate(layout=layout, seed=seed)
    rng = np.random.default_rng(1000 * seed + 17 * rep + int(level * 100))
    kw = {"depth": 1.0, "spot_frac": 1.0, "coord_sd": 0.0}; kw[axis] = level
    sim, idx = perturb(base, rng, **kw)
    members = P.domain_members(sim, K, 3, rng)
    out = []
    tests = M.global_svg_tests(sim.counts, sim.coords)
    ref = A.global_reference(tests)
    for mode in ["majority", "cac_weighted_pruned"] + ["kmeans_smooth", "kmeans_pca"]:
        lab = members[mode][0] if mode in members else E.ensemble(members, sim.coords, K, mode, seed=rep)["labels"]
        for rule, kw in RULES.items():
            hit, _ = M.domain_svgs(sim.counts, lab, **kw)
            called = hit.any(1)
            out.append({"layout": layout, "seed": seed, "axis": axis, "level": level, "rep": rep, "approach": mode, "rule": rule,
                        "ari": ari(sim.domains, lab), "domain_svg": called.tolist(),
                        **{f"svg_{k}": v for k, v in A.precision_recall(called, sim.is_svg).items()}})
    for name in ["intersection_rescued", "acat"]:
        out.append({"layout": layout, "seed": seed, "axis": axis, "level": level, "rep": rep, "approach": f"global:{name}", "rule": "global",
                    "ari": np.nan, "domain_svg": ref[name].tolist(),
                    **{f"svg_{k}": v for k, v in A.precision_recall(ref[name], sim.is_svg).items()}})
    return out


def e5(args):
    layout, seed, b = args
    base = simulate(layout=layout, seed=seed)
    rng = np.random.default_rng(50000 + 100 * seed + b)
    sim, idx = perturb(base, rng, depth=0.8, spot_frac=0.8, coord_sd=0.3)
    members = P.domain_members(sim, K, 3, rng)
    res = {"layout": layout, "seed": seed, "b": b}
    for mode in ["majority", "cac_weighted_pruned"]:
        lab = E.ensemble(members, sim.coords, K, mode, seed=b)["labels"]
        for rule, kw in RULES.items():
            hit, _ = M.domain_svgs(sim.counts, lab, **kw)
            called = np.zeros(len(base.gene_class), bool); called[:] = hit.any(1)
            res[f"{mode}|{rule}"] = called.tolist()
    res["method_ari"] = {m: float(np.mean([ari(sim.domains, l) for l in v])) for m, v in members.items()}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/pilot"); ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--quick", action="store_true"); ap.add_argument("--only", default="1,4,5")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    datasets = DATASETS[:2] if args.quick else DATASETS
    B = 6 if args.quick else 20
    reps = 1 if args.quick else 3
    only = set(args.only.split(","))
    t0 = time.time()
    with Pool(args.workers) as pool:
        if "1" in only:
            res = pool.map(e1_e2_e3, datasets)
            for key in ["e1", "prune", "calib", "e2", "e3"]:
                pd.DataFrame([r for x in res for r in x[key]]).to_csv(out / f"{key}.csv", index=False)
            pd.DataFrame([x["agreement"] for x in res]).to_csv(out / "agreement.csv", index=False)
            (out / "method_scores.json").write_text(json.dumps([x["scores"] for x in res], indent=1))
            print("E1-E3 done", round(time.time() - t0), flush=True)
        if "4" in only:
            jobs = [(l, s, ax, lv, r) for (l, s) in datasets for ax, levels in AXES.items() for lv in levels for r in range(reps)]
            res = pool.map(e4, jobs, chunksize=1)
            df = pd.DataFrame([r for x in res for r in x])
            # stability: Jaccard of each perturbed SVG set to the unperturbed set of the same approach/dataset
            base = df[((df.axis == "depth") & (df.level == 1.0) & (df.rep == 0))].groupby(["layout", "seed", "approach", "rule"]).domain_svg.first()
            df["jaccard_to_unperturbed"] = [A.jaccard(np.array(s), np.array(base[(l, sd, a, ru)]))
                                            for s, l, sd, a, ru in zip(df.domain_svg, df.layout, df.seed, df.approach, df.rule)]
            df.drop(columns="domain_svg").to_csv(out / "e4_robustness.csv", index=False)
            print("E4 done", round(time.time() - t0), flush=True)
        if "5" in only:
            jobs = [(l, s, b) for (l, s) in datasets for b in range(B)]
            res = pool.map(e5, jobs, chunksize=1)
            rows, rank_rows = [], []
            for (l, s) in datasets:
                sub = [x for x in res if x["layout"] == l and x["seed"] == s]
                sim = simulate(layout=l, seed=s)
                for mode in [f"{a}|{r}" for a in ["majority", "cac_weighted_pruned"] for r in RULES]:
                    Cm = np.array([x[mode] for x in sub])
                    for thr in [0.5, 0.6, 0.7, 0.8, 0.9]:
                        pi, sel, bound = A.stability_selection(Cm, max(thr, 0.51))
                        rows.append({"layout": l, "seed": s, "mode": mode, "pi_thr": thr, "mb_bound_EV": bound,
                                     "observed_FP": int((sel & ~sim.is_svg).sum()),
                                     "single_run_mean_FP": float(np.mean([(np.array(c) & ~sim.is_svg).sum() for c in Cm])),
                                     **A.precision_recall(sel, sim.is_svg)})
                    pi = Cm.mean(0)
                    np.save(out / f"pi_{l}_{s}_{mode.replace('|', '_')}.npy", pi)
                # benchmark-level uncertainty: which method ranks first in each replicate?
                for x in sub:
                    best = max(x["method_ari"], key=x["method_ari"].get)
                    rank_rows.append({"layout": l, "seed": s, "b": x["b"], "best": best, **{f"ari_{m}": v for m, v in x["method_ari"].items()}})
            pd.DataFrame(rows).to_csv(out / "e5_stability_selection.csv", index=False)
            pd.DataFrame(rank_rows).to_csv(out / "e5_rank_uncertainty.csv", index=False)
            truth = {f"{l}_{s}": simulate(layout=l, seed=s).is_svg.tolist() for (l, s) in datasets}
            (out / "truth_svg.json").write_text(json.dumps(truth))
            print("E5 done", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
