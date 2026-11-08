"""Aggregate v2 runs: summaries, paired KAN-vs-MLP statistics, ranks, figures.

Usage: python scripts/analyze_v2.py --runs results/v2/main [--audit results/v2/audit.json] [--sweep results/v2/sample_size]

Primary error (lower is better):
  regression     inside-test RMSE against the noise-free truth (synthetic tasks) or
                 the measured target (biology; identical to plain RMSE there). Biological
                 targets are in training-SD units; synthetic targets in their own units.
  classification 1 - macro-F1 on the inside test split
Paired comparisons use, within each (task, seed, depth, budget) block, the KAN
basis and the MLP activation chosen by *validation loss* (never test data), so
the comparison is the one a practitioner could actually make.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

MODELS = ["mlp-tanh", "mlp-relu", "mlp-silu", "kan-spline", "kan-fourier", "kan-rbf"]
COLORS = {"mlp-tanh": "#9ecae1", "mlp-relu": "#4292c6", "mlp-silu": "#08519c",
          "kan-spline": "#fdae6b", "kan-fourier": "#e6550d", "kan-rbf": "#a63603"}
SYNTH = ["linear", "additive", "interaction", "waves", "radial", "kink"]


def load_runs(folder):
    rows = []
    for p in sorted((Path(folder) / "runs").glob("*.json")):
        r = json.loads(p.read_text())
        row = {k: r[k] for k in ["task", "seed", "depth", "budget", "model", "parameters", "budget_relative_error",
                                  "macs_per_sample", "search_seconds", "task_type"]}
        row["n_train"] = r.get("n_train")
        s = r["selected"]
        row.update(lr=s["lr"], val_loss=s["val_loss"], epochs=s["epochs"], stop_reason=s["stop_reason"], fit_seconds=s["fit_seconds"])
        row["n_diverged"] = sum(1 for c in r["candidates"] if "error" in c)
        for split, m in r["metrics"].items():
            for k, v in m.items():
                row[f"{split}_{k}"] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    df["family"] = np.where(df.model.str.startswith("kan"), "KAN", "MLP")
    tiers = df.groupby("task").budget.agg(["min", "max"])
    df["tier"] = np.where(df.budget == df.task.map(tiers["min"]), "small", "large")
    for split in ["inside", "outside"]:
        if f"{split}_rmse" in df:
            err = df[f"{split}_clean_rmse"].copy()
            if f"{split}_macro_f1" in df:
                cls = df.task_type == "classification"
                err[cls] = 1 - df.loc[cls, f"{split}_macro_f1"]
            df[f"{split}_error"] = err
    return df


def ci95(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan, np.nan
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean() - h, x.mean() + h


def holm(p):
    p = np.asarray(p, float); out = np.full_like(p, np.nan); ok = np.isfinite(p)
    q = p[ok]; order = np.argsort(q); m = len(q)
    adj = np.maximum.accumulate((m - np.arange(m)) * q[order]); adj = np.minimum(adj, 1)
    tmp = np.empty(m); tmp[order] = adj; out[ok] = tmp
    return out


def selected_pairs(df, split="inside"):
    """Per block: validation-selected KAN vs validation-selected MLP."""
    rows = []
    key = ["task", "seed", "depth", "budget", "tier"] + (["n_train"] if df.n_train.notna().any() else [])
    for k, g in df.groupby(key, dropna=False):
        if f"{split}_error" not in g or g[f"{split}_error"].isna().all():
            continue
        kan = g[g.family == "KAN"].sort_values("val_loss").iloc[0]
        mlp = g[g.family == "MLP"].sort_values("val_loss").iloc[0]
        rec = dict(zip(key, k))
        rec.update(task_type=g.task_type.iloc[0], kan_model=kan.model, mlp_model=mlp.model,
                   kan_error=kan[f"{split}_error"], mlp_error=mlp[f"{split}_error"],
                   kan_params=kan.parameters, mlp_params=mlp.parameters,
                   kan_seconds=kan.search_seconds, mlp_seconds=mlp.search_seconds)
        rec["diff"] = rec["kan_error"] - rec["mlp_error"]
        rec["log_ratio"] = np.log(rec["kan_error"] / rec["mlp_error"]) if rec["mlp_error"] > 0 and rec["kan_error"] > 0 else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def paired_summary(pairs, by=("task", "depth", "tier")):
    out = []
    for k, g in pairs.groupby(list(by)):
        d = g["diff"].values
        lo, hi = ci95(d)
        try:
            p = stats.wilcoxon(d).pvalue if len(d) >= 5 and np.any(d != 0) else np.nan
        except ValueError:
            p = np.nan
        rec = dict(zip(by, k))
        rec.update(n=len(d), kan_error=g.kan_error.mean(), mlp_error=g.mlp_error.mean(), mean_diff=d.mean(),
                   ci_low=lo, ci_high=hi, kan_wins=int((d < 0).sum()), mean_log_ratio=g.log_ratio.mean(), p_wilcoxon=p,
                   kan_choice=g.kan_model.mode().iloc[0], mlp_choice=g.mlp_model.mode().iloc[0])
        out.append(rec)
    res = pd.DataFrame(out)
    res["p_holm"] = holm(res.p_wilcoxon.values)
    return res


def task_headline(pairs, seed_boot=4000):
    """Per task, pooled over depth and budget: mean log(KAN/MLP) with a 95 % CI
    from a bootstrap that resamples replicate seeds (the independent unit)."""
    rng = np.random.default_rng(0); out = []
    for t, g in pairs.groupby("task"):
        per_seed = g.groupby("seed").log_ratio.mean()
        boots = [per_seed.sample(len(per_seed), replace=True, random_state=int(rng.integers(1e9))).mean() for _ in range(seed_boot)]
        out.append({"task": t, "blocks": len(g), "kan_wins": int((g["diff"] < 0).sum()), "mean_log_ratio": g.log_ratio.mean(),
                    "ci_low": float(np.quantile(boots, 0.025)), "ci_high": float(np.quantile(boots, 0.975)),
                    "kan_error": g.kan_error.mean(), "mlp_error": g.mlp_error.mean()})
    return pd.DataFrame(out)


def friedman(df, split="inside"):
    """Six-model Friedman test over blocks (task, depth, tier), seeds averaged."""
    m = df.groupby(["task", "depth", "tier", "model"])[f"{split}_error"].mean().unstack("model")[MODELS].dropna()
    ranks = m.rank(axis=1)
    stat, p = stats.friedmanchisquare(*[m[c] for c in MODELS])
    k, n = len(MODELS), len(m)
    cd = 2.850 * np.sqrt(k * (k + 1) / (6 * n))  # Nemenyi, alpha = 0.05, k = 6
    return ranks, {"chi2": float(stat), "p": float(p), "blocks": int(n), "nemenyi_cd": float(cd),
                   "mean_rank": ranks.mean().round(3).to_dict()}


def fmt(x, nd=3):
    return "" if pd.isna(x) else f"{x:.{nd}f}"


def figures(df, pairs, ranks, audit, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    # 1. mean rank per task (heatmap)
    r = ranks.groupby(level="task").mean()
    order = [t for t in SYNTH if t in r.index] + sorted([t for t in r.index if t not in SYNTH])
    r = r.loc[order]
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(r) + 1.4))
    im = ax.imshow(r.values, cmap="RdYlGn_r", vmin=1, vmax=6, aspect="auto")
    ax.set_xticks(range(len(MODELS))); ax.set_xticklabels(MODELS, rotation=30, ha="right")
    ax.set_yticks(range(len(r))); ax.set_yticklabels(r.index)
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            ax.text(j, i, f"{r.values[i, j]:.1f}", ha="center", va="center", fontsize=8)
    ax.axhline(len([t for t in SYNTH if t in r.index]) - 0.5, color="k", lw=1)
    ax.set_title("Mean rank of inside-test error (1 = best); seed-averaged errors, ranked per depth x budget block", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.03); fig.tight_layout(); fig.savefig(outdir / "rank_heatmap.png", dpi=160); plt.close(fig)

    # 2. paired log-ratio by task (validation-selected KAN vs MLP)
    for split, pr in [("inside", pairs["inside"]), ("outside", pairs.get("outside"))]:
        if pr is None or pr.empty:
            continue
        tasks_ = [t for t in order if t in set(pr.task)]
        fig, ax = plt.subplots(figsize=(8, 0.38 * len(tasks_) + 1.5))
        for i, t in enumerate(tasks_):
            g = pr[pr.task == t]
            for (depth, tier), gg in g.groupby(["depth", "tier"]):
                off = {(1, "small"): -0.24, (1, "large"): -0.08, (2, "small"): 0.08, (2, "large"): 0.24}[(depth, tier)]
                v = gg.log_ratio.dropna()
                if len(v) == 0:
                    continue
                lo, hi = ci95(v)
                ax.errorbar(v.mean(), i + off, xerr=[[v.mean() - lo], [hi - v.mean()]] if np.isfinite(lo) else None,
                            fmt="o" if depth == 1 else "s", color="#555" if tier == "small" else "#d62728", ms=4, capsize=2)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(range(len(tasks_))); ax.set_yticklabels(tasks_); ax.invert_yaxis()
        ax.set_xlabel("log(error KAN / error MLP)   < 0: KAN better")
        ax.set_title(f"Validation-selected KAN vs MLP, {split} test (mean, 95% CI over seeds)\n"
                     "circle = 1 hidden layer, square = 2; grey = small budget, red = large budget", fontsize=9)
        fig.tight_layout(); fig.savefig(outdir / f"paired_log_ratio_{split}.png", dpi=160); plt.close(fig)

    # 3. structure fingerprint vs KAN advantage (regression tasks)
    if audit:
        pr = pairs["inside"]
        g = pr.groupby("task").log_ratio.mean()
        pts = [(t, audit[t]["interaction"], audit[t]["nonlinearity"], g[t]) for t in g.index if t in audit]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, idx, lab in [(axes[0], 1, "interaction  (R² flexible − R² additive)"), (axes[1], 2, "nonlinearity  (R² additive − R² linear)")]:
            for t, *vals in pts:
                ax.scatter(vals[idx - 1], vals[2], color="#e6550d" if t not in SYNTH else "#3182bd")
                ax.annotate(t, (vals[idx - 1], vals[2]), fontsize=7, xytext=(3, 2), textcoords="offset points")
            rho, p = stats.spearmanr([v[idx] for v in pts], [v[3] for v in pts])
            ax.axhline(0, color="k", lw=0.6); ax.set_xlabel(lab); ax.set_ylabel("mean log(KAN/MLP error)")
            ax.set_title(f"Spearman rho = {rho:.2f} (p = {p:.2f}, n = {len(pts)})", fontsize=9)
        fig.suptitle("Does audited structure predict where KANs help?  blue = synthetic control, orange = biology", fontsize=9)
        fig.tight_layout(); fig.savefig(outdir / "structure_vs_advantage.png", dpi=160); plt.close(fig)

    # 4. cost
    c = df.groupby(["tier", "model"]).search_seconds.median().unstack("model")[MODELS]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = np.arange(len(MODELS))
    for i, tier in enumerate(["small", "large"]):
        if tier in c.index:
            ax.bar(x + (i - 0.5) * 0.38, c.loc[tier].values, 0.38, label=f"{tier} budget", color=["#9ecae1", "#3182bd"][i])
    ax.set_xticks(x); ax.set_xticklabels(MODELS, rotation=20); ax.set_ylabel("median search seconds (3 LRs)"); ax.set_yscale("log")
    ax.legend(); ax.set_title("Training cost at matched parameter count (NumPy engine, 1 thread)")
    fig.tight_layout(); fig.savefig(outdir / "cost.png", dpi=160); plt.close(fig)


def sweep_figure(df, outdir):
    tasks_ = list(dict.fromkeys(df.task))
    fig, axes = plt.subplots(1, len(tasks_), figsize=(3.2 * len(tasks_), 3.2), sharey=False)
    for ax, t in zip(np.atleast_1d(axes), tasks_):
        g = df[df.task == t].groupby(["model", "n_train"]).inside_error.agg(["mean", "std"]).reset_index()
        for m in MODELS:
            gg = g[g.model == m]
            ax.errorbar(gg.n_train, gg["mean"], yerr=gg["std"], label=m, color=COLORS[m], marker="o", ms=3,
                        ls="-" if m.startswith("kan") else "--", capsize=2)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(t); ax.set_xlabel("training samples")
    np.atleast_1d(axes)[0].set_ylabel("inside error (noise-free RMSE)")
    np.atleast_1d(axes)[-1].legend(fontsize=7)
    fig.suptitle("Sample-size sweep (2 hidden layers, 1,024 parameters, 3 seeds)", fontsize=10)
    fig.tight_layout(); fig.savefig(outdir / "sample_size.png", dpi=160); plt.close(fig)


def markdown_tables(df, paired_in, paired_out, fr, outdir, heads=None, base=None, audit=None):
    lines = []
    if heads:
        lines.append("### Headline: validation-selected KAN vs MLP per task (pooled over depth and budget)\n")
        lines.append("ratio = exp(mean log(KAN error / MLP error)); < 1 means KAN better. CI: bootstrap over replicate seeds. "
                     "Baselines: RMSE of predicting the training mean / of ridge regression (cosinor for circadian tasks).\n")
        lines.append("| Task | Audited structure | Inside ratio (95% CI) | KAN wins | KAN / MLP inside error | Baseline mean / ridge | Outside ratio (95% CI) | KAN wins | KAN / MLP outside error | Baseline mean / ridge |")
        lines.append("|---|---|---|---:|---|---|---|---:|---|---|")
        hi = heads["inside"].set_index("task"); ho = heads.get("outside"); ho = ho.set_index("task") if ho is not None else None
        order = [t for t in SYNTH if t in hi.index] + sorted([t for t in hi.index if t not in SYNTH])
        def bl(t, split):
            if base is None:
                return ""
            b = base[(base.task == t) & (base.split == split)]
            if b.empty:
                return ""
            txt = f"{b['mean'].mean():.3f} / {b['ridge'].mean():.3f}"
            if "cosinor" in b and b["cosinor"].notna().any():
                txt += f" (cosinor {b['cosinor'].mean():.3f})"
            return txt
        for t in order:
            r = hi.loc[t]
            verdict = "; ".join(audit[t]["verdict"]) if audit and t in audit else ""
            cells = [t, verdict, f"{np.exp(r.mean_log_ratio):.3f} ({np.exp(r.ci_low):.3f}–{np.exp(r.ci_high):.3f})",
                     f"{int(r.kan_wins)}/{int(r.blocks)}", f"{r.kan_error:.3f} / {r.mlp_error:.3f}", bl(t, "inside")]
            if ho is not None and t in ho.index:
                o = ho.loc[t]
                cells += [f"{np.exp(o.mean_log_ratio):.3f} ({np.exp(o.ci_low):.3f}–{np.exp(o.ci_high):.3f})", f"{int(o.kan_wins)}/{int(o.blocks)}",
                          f"{o.kan_error:.3f} / {o.mlp_error:.3f}", bl(t, "outside")]
            else:
                cells += ["—", "", "", ""]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    lines.append("### Validation-selected KAN vs MLP, inside test\n")
    lines.append("| Task | Depth | Budget | n | KAN error | MLP error | KAN − MLP (95% CI) | KAN wins | Holm p | Usual picks |")
    lines.append("|---|---:|---|---:|---:|---:|---|---:|---:|---|")
    for _, r in paired_in.iterrows():
        lines.append(f"| {r.task} | {r.depth} | {r.tier} | {r.n} | {fmt(r.kan_error)} | {fmt(r.mlp_error)} | "
                     f"{r.mean_diff:+.3f} ({fmt(r.ci_low)}, {fmt(r.ci_high)}) | {r.kan_wins}/{r.n} | {fmt(r.p_holm)} | {r.kan_choice} / {r.mlp_choice} |")
    if paired_out is not None and len(paired_out):
        lines.append("\n### Validation-selected KAN vs MLP, outside (extrapolation) test\n")
        lines.append("| Task | Depth | Budget | n | KAN error | MLP error | KAN − MLP (95% CI) | KAN wins | Holm p |")
        lines.append("|---|---:|---|---:|---:|---:|---|---:|---:|")
        for _, r in paired_out.iterrows():
            lines.append(f"| {r.task} | {r.depth} | {r.tier} | {r.n} | {fmt(r.kan_error)} | {fmt(r.mlp_error)} | "
                         f"{r.mean_diff:+.3f} ({fmt(r.ci_low)}, {fmt(r.ci_high)}) | {r.kan_wins}/{r.n} | {fmt(r.p_holm)} |")
    lines.append("\n### Six-model mean error by task (inside test; mean over seeds, depths and budgets)\n")
    m = df.groupby(["task", "model"]).inside_error.mean().unstack("model")[MODELS]
    lines.append("| Task | " + " | ".join(MODELS) + " |"); lines.append("|---|" + "---:|" * len(MODELS))
    order = [t for t in SYNTH if t in m.index] + sorted([t for t in m.index if t not in SYNTH])
    for t in order:
        row = m.loc[t]; best = row.min()
        cells = [f"**{v:.3f}**" if np.isclose(v, best) else f"{v:.3f}" for v in row]
        lines.append(f"| {t} | " + " | ".join(cells) + " |")
    lines.append(f"\nFriedman test over {fr['blocks']} blocks: chi² = {fr['chi2']:.1f}, p = {fr['p']:.2g}; "
                 f"Nemenyi critical difference = {fr['nemenyi_cd']:.2f} ranks.\n")
    lines.append("| Model | Mean rank |"); lines.append("|---|---:|")
    for k, v in sorted(fr["mean_rank"].items(), key=lambda kv: kv[1]):
        lines.append(f"| {k} | {v:.2f} |")
    (outdir / "tables.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True); ap.add_argument("--audit"); ap.add_argument("--sweep"); ap.add_argument("--baselines")
    args = ap.parse_args()
    out = Path(args.runs)
    df = load_runs(out)
    df.to_csv(out / "metrics.csv", index=False)
    pairs = {"inside": selected_pairs(df, "inside")}
    if "outside_error" in df:
        pairs["outside"] = selected_pairs(df[df.outside_error.notna()], "outside")
    for k, v in pairs.items():
        v.to_csv(out / f"paired_{k}.csv", index=False)
    paired_in = paired_summary(pairs["inside"]); paired_in.to_csv(out / "paired_summary_inside.csv", index=False)
    paired_out = paired_summary(pairs["outside"]) if "outside" in pairs and len(pairs["outside"]) else None
    if paired_out is not None:
        paired_out.to_csv(out / "paired_summary_outside.csv", index=False)
    ranks, fr = friedman(df)
    overall = pairs["inside"].groupby("task_type").agg(n=("diff", "size"), kan_wins=("diff", lambda d: int((d < 0).sum())),
                                                        mean_log_ratio=("log_ratio", "mean"))
    summary = {"friedman": fr, "overall": overall.reset_index().to_dict(orient="records"),
               "runs": int(len(df)), "stop_reasons": df.stop_reason.value_counts().to_dict(),
               "max_budget_error": float(df.budget_relative_error.max()),
               "diverged_candidates": int(df.n_diverged.sum()),
               "kan_over_mlp_time": float(df[df.family == "KAN"].search_seconds.median() / df[df.family == "MLP"].search_seconds.median())}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    audit = json.loads(Path(args.audit).read_text()) if args.audit else None
    figures(df, pairs, ranks, audit, out / "figures")
    heads = {k: task_headline(v) for k, v in pairs.items() if len(v)}
    for k, v in heads.items():
        v.to_csv(out / f"headline_{k}.csv", index=False)
    base = pd.read_csv(args.baselines) if args.baselines and Path(args.baselines).exists() else None
    markdown_tables(df, paired_in, paired_out, fr, out, heads, base, audit)
    if args.sweep and (Path(args.sweep) / "runs").exists():
        sw = load_runs(args.sweep); sw.to_csv(Path(args.sweep) / "metrics.csv", index=False)
        sweep_figure(sw, out / "figures")
        sp = selected_pairs(sw, "inside"); sp.to_csv(Path(args.sweep) / "paired_inside.csv", index=False)
        paired_summary(sp, by=("task", "n_train")).to_csv(Path(args.sweep) / "paired_summary.csv", index=False)
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
