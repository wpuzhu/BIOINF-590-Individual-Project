"""Validation of the COMBINED framework (v1 + v2) on held-out simulations.

For each tissue, every approach runs on the same data so comparisons are paired.

Gene level (domain-specific SVGs)
  Kang strict / Kang liberal           reference pipeline, one run, same counts
  v1 stability                         pilot method: perturbed replicates, liberal SpaGCN rule, same counts, pi >= 0.6
  single split (+/- gate)              ablation: one count split, no stability
  SplitStab strict  (v2)               B replicates, strict NB rule, gate, NC threshold
  SplitStab liberal (combined)         B replicates, liberal NB rule, gate, NC threshold
  ... and each SplitStab rule without the gate, and without core spots
Global SVGs
  intersection (Kang), ACAT on full data (v1), stability-selected ACAT on held-out halves (combined)
Uncertainty
  spot level: PCE confidence and replicate agreement vs correctness, true mixing (membership), distance to boundary
  gene level: pi vs truth and effect size; modal-domain correctness
  method level: per-replicate ARI -> P(best), and how often pruning keeps each method

Usage: python scripts/run_combined.py --out results/combined --workers 2 --B 20
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import adjusted_rand_score as ari

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stbench import ensemble as E, methods as M, svg_aggregate as A, workflow as W  # noqa: E402
from stbench.simulate import simulate, thin_counts  # noqa: E402

K = 6
METHODS = ["kmeans_pca", "gmm_pca", "kmeans_smooth", "ward_spatial", "spectral_spatial"]
SIGNAL = [(l, s) for l in ["layers", "blobs"] for s in [4, 5, 6]]
NULL = [21, 22, 23, 24]


def methods():
    return {m: W.CPU_METHODS[m] for m in METHODS}


def boundary_distance(coords, labels):
    """Distance from each spot to the nearest spot with a different label (spot spacings)."""
    d = np.full(len(labels), np.inf)
    for l in np.unique(labels):
        other = coords[labels != l]
        if len(other):
            dist, _ = cKDTree(other).query(coords[labels == l])
            d[labels == l] = dist
    return d


def job(arg):
    kind, layout, seed, depth, B = arg
    if kind == "null":
        sim = simulate(seed=seed, n_domain_genes=0, n_gradient_genes=0)
    else:
        sim = simulate(layout=layout, seed=seed)
    rng = np.random.default_rng(1000 * seed + int(depth * 100))
    counts = thin_counts(sim.counts, depth, rng) if depth < 1 else sim.counts
    ctx = W.Context(counts, sim.coords)
    truth = sim.is_svg
    calls = {}
    t0 = time.time()
    calls["Kang strict"] = W.paper_pipeline(ctx, K, methods(), rng, strict=True)[0]
    calls["Kang liberal"] = W.paper_pipeline(ctx, K, methods(), rng, strict=False)[0]
    v1 = W.v1_stability(ctx, K, methods(), B=B, rng=rng)
    calls["v1 stability (pi>=0.6)"] = v1["selected"]
    for r, v in W.single_split(ctx, K, methods(), rng=rng).items():
        calls[f"single split {r.replace('_nogate', '')}{' no gate' if r.endswith('_nogate') else ''}"] = v
    # full-data PCE (reference labels + confidence)
    m0 = W.run_members(ctx, K, methods(), 3, rng)
    pce = E.pce(m0, ctx.coords, K, seed=seed)
    ss = W.splitstab(ctx, K, methods(), B=B, rng=rng, reference=pce["labels"], record_members=(kind == "signal"))
    for r, v in ss["by_rule"].items():
        calls[f"SplitStab {r}"] = v["selected"]
    for r, v in ss["by_rule_nogate"].items():
        calls[f"SplitStab {r} no gate"] = v["selected"]
    # global
    T = M.global_svg_tests(counts, sim.coords); R = A.global_reference(T)
    gcalls = {"global intersection (Kang)": R["intersection_rescued"], "global ACAT full data (v1)": R["acat"],
              "global ACAT split + stability (combined)": ss["global"]["selected"]}
    rows = []
    for name, c in {**calls, **gcalls}.items():
        pr = A.precision_recall(c, truth)
        rows.append({"kind": kind, "layout": layout, "seed": seed, "depth": depth, "approach": name,
                     "n_called": int(c.sum()), "false_calls": int((c & ~truth).sum()), **pr})
    info = {"kind": kind, "layout": layout, "seed": seed, "depth": depth, "seconds": round(time.time() - t0),
            "thresholds": {r: v["threshold"] for r, v in ss["by_rule"].items()},
            "fdp_tables": {r: v["fdp_table"] for r, v in ss["by_rule"].items()},
            "v1_mb_bound": v1["mb_bound"]}
    unc = None
    if kind == "signal":
        # spot level
        ref_true = E.align(sim.domains, pce["labels"], K)
        correct = ref_true == sim.domains
        unc = {"spot": {"confidence": pce["confidence"], "agreement": ss["spot_agreement"], "stability": ss["spot_stability"], "correct": correct,
                        "membership": sim.meta["membership_max"], "boundary_dist": boundary_distance(sim.coords, sim.domains)},
               "gene": {}}
        # gene level (primary = liberal)
        pr = ss["by_rule"]["liberal"]
        # map reference domain -> true domain by majority overlap
        ref2true = np.array([np.bincount(sim.domains[pce["labels"] == d], minlength=K).argmax() if (pce["labels"] == d).any() else -1
                             for d in range(K)])
        unc["gene"] = {"pi": pr["pi"], "pi_strict": ss["by_rule"]["strict"]["pi"], "pi_v1": v1["pi"], "pi_global": ss["global"]["pi"],
                       "truth": truth, "gene_class": sim.gene_class, "logfc": sim.gene_logfc, "target": sim.gene_domain,
                       "modal_true": ref2true[pr["modal_domain"]], "purity": pr["domain_purity"], "pi_nc": pr["pi_nc"]}
        # method level
        mrows = []
        for b, recd in enumerate(ss["members_record"]):
            for m, lab in recd["members"].items():
                mrows.append({"b": b, "method": m, "ari": ari(sim.domains[recd["idx"]], lab), "kept": m in (recd["kept"] or [])})
        unc["methods"] = mrows
        unc["full_member_ari"] = {m: float(np.mean([ari(sim.domains, l) for l in v])) for m, v in m0.items()}
        unc["pce_ari"] = float(ari(sim.domains, pce["labels"]))
    return rows, info, unc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/combined"); ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--B", type=int, default=20)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    jobs = [("null", "layers", s, 1.0, args.B) for s in NULL]
    jobs += [("signal", l, s, d, args.B) for d in [1.0, 0.25] for l, s in SIGNAL]
    t0 = time.time(); rows, infos = [], []
    with Pool(args.workers) as pool:
        for r, info, unc in pool.imap_unordered(job, jobs):
            rows += r; infos.append(info)
            if unc is not None:
                tag = f"{info['layout']}_{info['seed']}_{info['depth']}"
                np.savez_compressed(out / f"unc_{tag}.npz", **{f"spot_{k}": v for k, v in unc["spot"].items()},
                                    **{f"gene_{k}": v for k, v in unc["gene"].items()})
                pd.DataFrame(unc["methods"]).to_csv(out / f"methods_{tag}.csv", index=False)
                info["full_member_ari"] = unc["full_member_ari"]; info["pce_ari"] = unc["pce_ari"]
            pd.DataFrame(rows).to_csv(out / "calls.csv", index=False)
            (out / "info.json").write_text(json.dumps(infos, indent=1, default=float))
            print(info["kind"], info["layout"], info["seed"], info["depth"], info["seconds"], "s; total", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
