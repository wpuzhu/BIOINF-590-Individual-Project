"""Validation of the expanded methods on HELD-OUT simulations (seeds 4-9; the
pilot used 1-3, and the pruning rule was fixed on the pilot).

V1  Domain aggregation: majority (Kang 2025), STCC-average, EnSDD-weights,
    CAC weighted, PCE (pre-registered pruning), plus single methods.
V2  Double dipping: tissues with NO spatial signal. Every domain-SVG call is a
    false positive. Same-data tests vs count-split tests vs SplitStab.
V3  SplitStab vs the paper's pipeline and a liberal single run: precision,
    recall, achieved FDP vs the alpha = 0.1 target, calibration of pi.
V4  Global SVG aggregation: intersection (paper) vs ACAT vs Castl-style
    (rank, Cauchy + BY, negative-control FDP).
V5  Low depth (25 %): paper pipeline vs SplitStab.

Usage: python scripts/run_validation.py --out results/validation --workers 2 [--B 20]
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
from stbench import ensemble as E, methods as M, svg_aggregate as A, workflow as W  # noqa: E402
from stbench.domain_svg import spatial_gate, test_domains  # noqa: E402
from stbench.simulate import simulate, thin_counts  # noqa: E402
from stbench.splitting import count_split  # noqa: E402

K = 6
SEEDS = [(l, s) for l in ["layers", "blobs"] for s in [4, 5, 6]]
METHODS = ["kmeans_pca", "gmm_pca", "kmeans_smooth", "ward_spatial", "spectral_spatial"]


def methods():
    return {m: W.CPU_METHODS[m] for m in METHODS}


def ece(conf, correct, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if s.any():
            e += s.mean() * abs(correct[s].mean() - conf[s].mean())
    return float(e)


def v1(arg):
    layout, seed = arg
    sim = simulate(layout=layout, seed=seed); ctx = W.Context(sim.counts, sim.coords)
    members = W.run_members(ctx, K, methods(), 4, np.random.default_rng(seed))
    rows = [{"layout": layout, "seed": seed, "approach": m, "kind": "single", "ari": float(np.mean([ari(sim.domains, l) for l in v]))}
            for m, v in members.items()]
    for name, res in E.run_all_ensembles(members, ctx.coords, K, seed).items():
        rec = {"layout": layout, "seed": seed, "approach": name, "kind": "ensemble", "ari": ari(sim.domains, res["labels"]),
               "kept": ",".join(res.get("kept") or [])}
        if res.get("confidence") is not None:
            correct = E.align(sim.domains, res["labels"], K) == sim.domains
            rec["ece"] = ece(res["confidence"], correct)
            rec["acc_conf_ge_0.6"] = float(correct[res["confidence"] >= 0.6].mean()) if (res["confidence"] >= 0.6).any() else np.nan
            rec["acc_conf_lt_0.6"] = float(correct[res["confidence"] < 0.6].mean()) if (res["confidence"] < 0.6).any() else np.nan
            rec["share_core"] = float((res["confidence"] >= 0.6).mean())
        rows.append(rec)
    return rows


def v2(arg):
    seed, B = arg
    sim = simulate(seed=seed, n_domain_genes=0, n_gradient_genes=0)  # no spatial signal at all
    ctx = W.Context(sim.counts, sim.coords); rng = np.random.default_rng(seed)
    out = {"seed": seed}
    called, res = W.paper_pipeline(ctx, K, methods(), rng, strict=True); out["paper rule, same data (strict)"] = int(called.sum())
    called, _ = W.paper_pipeline(ctx, K, methods(), rng, strict=False); out["paper rule, same data (liberal)"] = int(called.sum())
    members = W.run_members(ctx, K, methods(), 2, rng)
    pce = E.run_all_ensembles(members, ctx.coords, K, seed)["PCE (proposed)"]
    hit, _, _ = test_domains(sim.counts, pce["labels"], pce["confidence"]); out["v2 rule, same data"] = int(hit.any(1).sum())
    X1, X2 = count_split(sim.counts, 0.5, rng)
    c1 = W.Context(X1, sim.coords); m1 = W.run_members(c1, K, methods(), 2, rng)
    p1 = E.run_all_ensembles(m1, c1.coords, K, seed)["PCE (proposed)"]
    hit, _, _ = test_domains(X2, p1["labels"], p1["confidence"]); out["v2 rule, count split"] = int(hit.any(1).sum())
    out["v2 rule, count split + spatial gate"] = int((hit.any(1) & spatial_gate(X2, sim.coords)[0]).sum())
    hitw, _, _ = test_domains(X2, p1["labels"], p1["confidence"], method="wilcoxon"); out["Wilcoxon rule, count split"] = int(hitw.any(1).sum())
    ss0 = W.splitstab(ctx, K, methods(), B=B, rng=rng, gate=False)
    out["SplitStab without spatial gate"] = int(ss0["selected"].sum())
    ss = W.splitstab(ctx, K, methods(), B=B, rng=rng)
    out["SplitStab (selected)"] = int(ss["selected"].sum()); out["SplitStab threshold"] = ss["threshold"]
    return out


def v3(arg):
    layout, seed, B, depth = arg
    sim = simulate(layout=layout, seed=seed)
    rng = np.random.default_rng(100 + seed)
    counts = thin_counts(sim.counts, depth, rng)
    ctx = W.Context(counts, sim.coords)
    rec = {"layout": layout, "seed": seed, "depth": depth}
    called, _ = W.paper_pipeline(ctx, K, methods(), rng, strict=True)
    for k, v in A.precision_recall(called, sim.is_svg).items():
        rec[f"paper_strict_{k}"] = v
    called, _ = W.paper_pipeline(ctx, K, methods(), rng, strict=False)
    for k, v in A.precision_recall(called, sim.is_svg).items():
        rec[f"paper_liberal_{k}"] = v
    ss = W.splitstab(ctx, K, methods(), B=B, rng=rng)
    for k, v in A.precision_recall(ss["selected"], sim.is_svg).items():
        rec[f"splitstab_{k}"] = v
    rec["splitstab_threshold"] = ss["threshold"]
    for t in [0.6, 0.7, 0.8, 0.9]:
        rec[f"splitstab_pi{t}_recall"] = A.precision_recall(ss["pi"] >= t, sim.is_svg)["recall"]
        rec[f"splitstab_pi{t}_precision"] = A.precision_recall(ss["pi"] >= t, sim.is_svg)["precision"]
    # domain assignment accuracy of selected domain genes
    sel_dom = ss["selected"] & (sim.gene_class == "domain")
    rec["n_pi"] = 0
    return rec, ss["pi"].tolist(), sim.is_svg.tolist(), ss["fdp_table"]


def v4(arg):
    layout, seed = arg
    sim = simulate(layout=layout, seed=seed)
    T = M.global_svg_tests(sim.counts, sim.coords); R = A.global_reference(T)
    C = A.castl_like(T, sim.counts, sim.coords, M.global_svg_tests, rng=np.random.default_rng(seed))
    rows = []
    for name, sel in [("intersection (Kang 2025)", R["intersection_rescued"]), ("ACAT + BH (ours)", R["acat"]),
                      ("Castl-like rank_agg", C["rank_agg"]), ("Castl-like Cauchy + BY", C["pval_agg"]), ("Castl-like NC-FDP", C["fdp_agg"])]:
        rows.append({"layout": layout, "seed": seed, "reference": name, **A.precision_recall(sel, sim.is_svg)})
    # leave-one-out references: how much does including a method in its own reference flatter it?
    for t in ["morans_i", "gearys_c", "sparkx_like"]:
        own = M.bh(T[t][1]) < 0.05
        loo = A.leave_one_out_reference(T, t)
        rows.append({"layout": layout, "seed": seed, "reference": f"LOO check: {t}",
                     "recall_vs_full_intersection": A.precision_recall(own, R["intersection"])["precision"],
                     "recall_vs_loo_acat": A.precision_recall(own, loo)["precision"],
                     "precision_vs_truth": A.precision_recall(own, sim.is_svg)["precision"]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/validation"); ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--B", type=int, default=20); ap.add_argument("--only", default="1,2,3,4,5")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    only = set(args.only.split(",")); t0 = time.time()
    with Pool(args.workers) as pool:
        if "1" in only:
            pd.DataFrame([r for x in pool.map(v1, SEEDS) for r in x]).to_csv(out / "v1_domains.csv", index=False)
            print("V1", round(time.time() - t0), flush=True)
        if "4" in only:
            pd.DataFrame([r for x in pool.map(v4, SEEDS) for r in x]).to_csv(out / "v4_global.csv", index=False)
            print("V4", round(time.time() - t0), flush=True)
        if "2" in only:
            pd.DataFrame(pool.map(v2, [(s, max(6, args.B // 2)) for s in [21, 22, 23, 24]], chunksize=1)).to_csv(out / "v2_double_dipping.csv", index=False)
            print("V2", round(time.time() - t0), flush=True)
        jobs = []
        if "3" in only:
            jobs += [(l, s, args.B, 1.0) for l, s in SEEDS]
        if "5" in only:
            jobs += [(l, s, max(6, args.B // 2), d) for l, s in SEEDS for d in [0.5, 0.25]]
        if jobs:
            res = pool.map(v3, jobs, chunksize=1)
            pd.DataFrame([r[0] for r in res]).to_csv(out / "v3_splitstab.csv", index=False)
            pis = {f"{r[0]['layout']}_{r[0]['seed']}_{r[0]['depth']}": {"pi": r[1], "truth": r[2], "fdp_table": r[3]} for r in res}
            (out / "v3_pi.json").write_text(json.dumps(pis))
            print("V3/V5", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
