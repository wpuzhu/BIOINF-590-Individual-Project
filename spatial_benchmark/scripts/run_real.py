"""Run the real-data benchmark (DLPFC by default). Designed for Google Colab but
runs anywhere with anndata (+ huggingface_hub or pysodb for downloads).

Usage:
  python scripts/run_real.py --config configs/real_quick.json
  python scripts/run_real.py --config configs/real_full.json --out /content/drive/MyDrive/svg_bench

Each sample is checkpointed to <out>/<sample>/; re-running skips finished parts,
so a Colab disconnect costs at most one sample.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stbench import workflow as W  # noqa: E402
from stbench.real import evaluate as R  # noqa: E402
from stbench.real import io  # noqa: E402


def say(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def build_methods(cfg, log):
    methods = {m: W.CPU_METHODS[m] for m in cfg.get("cpu_methods", [])}
    if cfg.get("gpu_methods"):
        from stbench.real import gpu_methods as G
        ok, problems = G.available(cfg["gpu_methods"])
        for n, p in problems.items():
            msg = f"GPU method {n} unavailable: {p}"; log.append(msg); say(msg)
        methods.update(ok)
    return methods


def jdump(obj, path):
    Path(path).write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def run_sample(sample, cfg, methods, out):
    d = out / sample; d.mkdir(parents=True, exist_ok=True)
    done = d / "done.json"
    if done.exists():
        say(sample, "already done, skipping"); return
    log = []
    say(sample, "loading")
    if cfg.get("dataset", "dlpfc") == "sim":
        ctx, source = io.sim_context(sample), "simulated"
    else:
        adata, source = io.load_dlpfc(sample, cache_dir=cfg.get("cache_dir", "data"))
        ctx = io.to_context(adata, cfg.get("label_col"), max_genes=cfg.get("max_genes", 6000))
        del adata
    k = int(cfg.get("k") or len(ctx.extra["label_names"]))
    meta = {"sample": sample, "source": source, "k": k, **{x: ctx.extra[x] for x in ["count_source", "label_col", "n_spots", "n_genes", "label_names"]}}
    jdump(meta, d / "meta.json"); say(sample, meta)
    B = cfg.get("B", 10); exps = set(cfg.get("experiments", ["R1", "R2", "R3", "R5", "R6"]))
    seed = int("".join(ch for ch in sample if ch.isdigit()) or 0) % 10007

    # R1: domains
    rows, members, ens, scores = R.r1_domains(ctx, k, methods, cfg.get("n_reps", 3), seed, log)
    pd.DataFrame(rows).to_csv(d / "r1_domains.csv", index=False)
    jdump({m: {kk: float(v) for kk, v in sc.items()} for m, sc in scores.items()}, d / "method_scores.json")
    np.savez_compressed(d / "domain_labels.npz", truth=ctx.extra["labels"], coords=ctx.coords,
                        **{f"ens__{n}": e["labels"] for n, e in ens.items()},
                        **{f"conf__{n}": e["confidence"] for n, e in ens.items() if e.get("confidence") is not None},
                        **{f"single__{m}": v[0] for m, v in members.items()})
    kept = ens["PCE (proposed)"]["kept"]
    say(sample, "R1 done; PCE kept", kept)
    kept_methods = {m: methods[m] for m in kept}
    # R2/R3/R5 study the SVG step, not the domain-method choice: they use the two
    # best-scoring CPU methods so GPU cost stays bounded (configurable).
    cpu_ranked = sorted([m for m in scores if m in W.CPU_METHODS], key=lambda m: -scores[m]["score"])
    svg_methods = kept_methods if cfg.get("svg_methods", "cpu_kept") == "kept" else {m: methods[m] for m in cpu_ranked[:2]}
    reps = int(cfg.get("splitstab_reps", 2))

    # base SVG lists (also used by R4/R5)
    lists = {}
    # Kang pipeline: one run per method (all methods) -> majority -> strict SpaGCN rule
    called, _ = W.paper_pipeline(ctx, k, methods, np.random.default_rng(seed), strict=True)
    lists["Kang 2025 pipeline"] = called
    ss = W.splitstab(ctx, k, kept_methods, B=B, n_reps=reps, rng=np.random.default_rng(seed + 1), log=log,
                     progress=lambda b, n: say(sample, f"SplitStab {b}/{n}") if b % 5 == 0 or b == n else None)
    lists["SplitStab (proposed)"] = ss["selected"]
    pd.DataFrame({"gene": ctx.genes, "pi": ss["pi"], "selected": ss["selected"], "modal_domain": ss["modal_domain"],
                  "kang_pipeline": lists["Kang 2025 pipeline"]}).to_csv(d / "domain_svgs.csv", index=False)
    jdump({"threshold": ss["threshold"], "fdp_table": ss["fdp_table"], "mean_calls_per_replicate": ss["mean_calls_per_replicate"],
           "pi_nc": ss["pi_nc"]}, d / "splitstab.json")
    say(sample, f"SplitStab: {int(ss['selected'].sum())} genes at pi >= {ss['threshold']}; Kang pipeline: {int(called.sum())}")

    if "R6" in exps:
        rows6, glists, T = R.r6_global(ctx, seed)
        pd.DataFrame(rows6).to_csv(d / "r6_global.csv", index=False)
        lists.update(glists)
        pd.DataFrame({"gene": ctx.genes, **{k2: v for k2, v in glists.items()},
                      **{f"p_{t}": T[t][1] for t in ["morans_i", "gearys_c", "sparkx_like"]}}).to_csv(d / "global_svgs.csv", index=False)
    np.savez_compressed(d / "lists.npz", genes=np.array(ctx.genes), **{n.replace(" ", "_"): v for n, v in lists.items()})
    jdump({"names": list(lists)}, d / "list_names.json")
    if "R2" in exps:
        say(sample, "R2 double-dipping null"); jdump(R.r2_null(ctx, k, svg_methods, max(4, B // 2), seed + 2, log), d / "r2_null.json")
    if "R3" in exps:
        say(sample, "R3 spike-in")
        rows3, info, _ = R.r3_spikein(ctx, k, svg_methods, B, seed + 3, log)
        pd.DataFrame(rows3).to_csv(d / "r3_spikein.csv", index=False)
    if "R5" in exps:
        say(sample, "R5 perturbations")
        pd.DataFrame(R.r5_perturb(ctx, k, svg_methods, lists, max(4, B // 2), seed + 5, log=log)).to_csv(d / "r5_perturb.csv", index=False)
    jdump(log, d / "log.json")
    jdump({"finished": time.strftime("%Y-%m-%d %H:%M:%S")}, done)
    say(sample, "done")


def run_pairs(cfg, out):
    rows = []
    for a, b in cfg.get("pairs", []):
        fa, fb = out / a / "lists.npz", out / b / "lists.npz"
        if not (fa.exists() and fb.exists()):
            continue
        la, lb = np.load(fa, allow_pickle=True), np.load(fb, allow_pickle=True)
        names = [n for n in la.files if n != "genes" and n in lb.files]
        for r in R.r4_replication({n: la[n] for n in names}, {n: lb[n] for n in names}, list(la["genes"]), list(lb["genes"])):
            rows.append({"pair": f"{a}/{b}", **r})
    if rows:
        pd.DataFrame(rows).to_csv(out / "r4_replication.csv", index=False)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--out")
    ap.add_argument("--samples", help="comma-separated override")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if args.samples:
        cfg["samples"] = args.samples.split(",")
    out = Path(args.out or cfg.get("out", "results/real")); out.mkdir(parents=True, exist_ok=True)
    jdump(cfg, out / "config.json")
    log = []
    methods = build_methods(cfg, log)
    say("methods:", list(methods))
    jdump({"methods": list(methods), "log": log}, out / "methods.json")
    for s in cfg["samples"]:
        try:
            run_sample(s, cfg, methods, out)
        except Exception as e:
            say(f"{s} FAILED: {type(e).__name__}: {e}")
            (out / s).mkdir(exist_ok=True)
            jdump({"error": f"{type(e).__name__}: {e}"}, out / s / "error.json")
    if "R4" in cfg.get("experiments", ["R4"]):
        run_pairs(cfg, out)
    from report_real import build_report
    path = build_report(out)
    say("report:", path)


if __name__ == "__main__":
    main()
