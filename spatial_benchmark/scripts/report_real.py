"""Build a single self-contained HTML report from a run_real.py output folder.

Usage: python scripts/report_real.py results/real_quick
"""
import base64
import html
import io as _io
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
ENS_ORDER = ["majority (Kang 2025)", "STCC-average", "EnSDD-weights", "CAC weighted", "PCE (proposed)"]


def _img(fig):
    buf = _io.BytesIO(); fig.savefig(buf, format="png", dpi=150, bbox_inches="tight"); plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"/>'


def _table(df, floatfmt="{:.3f}"):
    if df is None or len(df) == 0:
        return "<p><em>not available</em></p>"
    return df.to_html(index=False, float_format=lambda v: floatfmt.format(v), border=0, classes="t", na_rep="—")


def _load(out, name, kind="csv"):
    frames = []
    for d in sorted(p for p in out.iterdir() if p.is_dir()):
        f = d / name
        if f.exists() and f.stat().st_size > 1:
            if kind == "csv":
                df = pd.read_csv(f); df.insert(0, "sample", d.name); frames.append(df)
            else:
                frames.append({"sample": d.name, **json.loads(f.read_text())})
    if kind == "csv":
        return pd.concat(frames, ignore_index=True) if frames else None
    return pd.DataFrame(frames) if frames else None


def section_domains(out):
    r1 = _load(out, "r1_domains.csv")
    if r1 is None:
        return "", {}
    piv = r1.pivot_table(index="sample", columns="approach", values="ari")
    order = [c for c in piv.columns if c not in ENS_ORDER] + [c for c in ENS_ORDER if c in piv.columns]
    piv = piv[order]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for i, c in enumerate(order):
        v = piv[c].dropna().values
        col = "#9ecae1" if c not in ENS_ORDER else ("#bdbdbd" if "majority" in c else ("#e6550d" if "PCE" in c else "#fdae6b"))
        ax.bar(i, v.mean(), color=col, width=0.7)
        ax.scatter(np.full(len(v), i) + np.linspace(-0.2, 0.2, len(v)), v, s=8, color="k", zorder=3)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=35, ha="right"); ax.set_ylabel("ARI vs manual layers")
    ax.set_title("Spatial domains: single methods (blue) and aggregation strategies (orange)")
    fig_html = _img(fig)
    stats = {"n_slices": int(len(piv))}
    if "PCE (proposed)" in piv and "majority (Kang 2025)" in piv:
        diff = (piv["PCE (proposed)"] - piv["majority (Kang 2025)"]).dropna()
        rng = np.random.default_rng(0)
        boots = [rng.choice(diff.values, len(diff)).mean() for _ in range(4000)] if len(diff) > 1 else [diff.mean()]
        stats.update(pce_minus_majority=float(diff.mean()), wins=int((diff > 0).sum()), ci=np.quantile(boots, [0.025, 0.975]).tolist())
        singles = [c for c in order if c not in ENS_ORDER]
        stats["pce_minus_best_single"] = float((piv["PCE (proposed)"] - piv[singles].max(axis=1)).mean()) if singles else np.nan
    # probability each approach is best (bootstrap over slices)
    rng = np.random.default_rng(1); best = []
    for _ in range(2000):
        s = piv.sample(len(piv), replace=True, random_state=int(rng.integers(1e9))).mean()
        best.append(s.idxmax())
    pbest = pd.Series(best).value_counts(normalize=True).rename("P(best)").reset_index().rename(columns={"index": "approach"})
    conf_cols = [c for c in ["ece", "share_core", "acc_core", "acc_noncore"] if c in r1]
    calib = r1[r1.kind == "ensemble"].groupby("approach")[conf_cols].mean().reset_index() if conf_cols else None
    kept = r1[r1.approach == "PCE (proposed)"][["sample", "kept"]]
    mean_tbl = piv.mean().rename("mean ARI").reset_index()
    text = f"""
<h2>1. Spatial domains (R1)</h2>
{fig_html}
<p>{_interp_domains(stats)}</p>
<h3>Mean ARI across slices</h3>{_table(mean_tbl)}
<h3>Uncertainty of the ranking: probability each approach is best (bootstrap over slices)</h3>{_table(pbest)}
<h3>Per-spot confidence (ensembles with a confidence output)</h3>
<p>ECE = expected calibration error (lower is better); "core" spots have confidence ≥ 0.6. A useful confidence has acc_core clearly above acc_noncore.</p>
{_table(calib)}
<h3>Methods kept by the pre-registered pruning rule</h3>{_table(kept, "{}")}
"""
    return text, stats


def _interp_domains(s):
    if "pce_minus_majority" not in s:
        return "Not enough results to compare aggregation strategies."
    lo, hi = s["ci"]
    verdict = ("improves on" if lo > 0 else "does not reliably differ from" if hi > 0 else "is worse than")
    txt = (f"Across {s['n_slices']} slices, the pruned co-association ensemble (PCE) {verdict} the majority vote of Kang et al. "
           f"(mean ΔARI = {s['pce_minus_majority']:+.3f}, 95% bootstrap CI {lo:+.3f} to {hi:+.3f}; better in {s['wins']}/{s['n_slices']} slices).")
    if np.isfinite(s.get("pce_minus_best_single", np.nan)):
        txt += (f" Relative to the best single method on each slice (known only in hindsight), PCE differs by "
                f"{s['pce_minus_best_single']:+.3f} ARI — the closer to zero, the more pruning recovers the oracle choice without labels.")
    return txt


def section_svgs(out):
    parts = []
    r2 = _load(out, "r2_null.json", "json")
    if r2 is not None:
        cols = [c for c in r2.columns if c not in ("sample",)]
        parts.append(f"""<h2>2. Double dipping: slice-calibrated null (R2)</h2>
<p>Counts are regenerated from each slice's own gene means, dispersions and library sizes, but with no spatial or domain structure.
<b>Every call here is a false positive.</b> Same-data pipelines (domains and tests on the same counts) show how many spurious
domain markers double dipping creates; the count-split rows and SplitStab should be near zero.</p>{_table(r2[['sample'] + cols], '{:.0f}')}""")
    r3 = _load(out, "r3_spikein.csv")
    if r3 is not None:
        agg = r3.drop(columns=["sample"]).groupby("approach").mean(numeric_only=True).reset_index()
        fig, ax = plt.subplots(figsize=(6, 3))
        fcols = [c for c in agg.columns if c.startswith("recall_fold_")]
        for _, row in agg.iterrows():
            ax.plot([c.replace("recall_fold_", "") for c in fcols], row[fcols].values, marker="o", label=row.approach)
        ax.set_xlabel("implanted fold change"); ax.set_ylabel("recall of spiked markers"); ax.set_ylim(0, 1.02); ax.legend(fontsize=7, frameon=False)
        parts.append(f"""<h2>3. Semi-synthetic spike-in on real slices (R3)</h2>
<p>Known markers are implanted into real low-autocorrelation genes inside one annotated layer. Recall is measured on the spiked genes;
the false-call rate is measured on the remaining low-autocorrelation genes, which are presumed (not known) to be non-spatial.</p>
{_img(fig)}{_table(agg)}""")
    r4 = out / "r4_replication.csv"
    if r4.exists():
        df = pd.read_csv(r4)
        agg = df.groupby("approach")[["n_a", "n_b", "jaccard", "replication_rate"]].mean().reset_index()
        parts.append(f"""<h2>4. Adjacent-section replication (R4)</h2>
<p>Sections from the same donor 10 µm apart should share most true SVGs. Higher Jaccard and replication rate mean a more reproducible list
(no ground truth needed). A list can replicate well by being short, so read this together with list sizes (n_a, n_b).</p>{_table(agg)}
<details><summary>per pair</summary>{_table(df)}</details>""")
    r4d = out / "r4_cross_donor.csv"
    if r4d.exists():
        df = pd.read_csv(r4d)
        agg = df.groupby("approach")[["n_a", "n_b", "jaccard", "replication_rate"]].mean().reset_index()
        parts.append(f"""<h3>4b. Cross-donor replication</h3>
<p>Same layer, different people. Lower than adjacent-section replication is expected (biological variation between donors);
this is the level of uncertainty that resampling one slice cannot capture.</p>{_table(agg)}""")
    r5 = _load(out, "r5_perturb.csv")
    if r5 is not None:
        agg = r5.groupby(["axis", "level", "approach"]).mean(numeric_only=True).reset_index()
        parts.append(f"""<h2>5. Robustness to perturbation on real slices (R5)</h2>
<p>Jaccard between the SVG list from a perturbed slice (thinned reads or dropped spots) and the list from the full slice.</p>{_table(agg)}""")
    r6 = _load(out, "r6_global.csv")
    if r6 is not None:
        agg = r6.groupby("approach").mean(numeric_only=True).reset_index()
        parts.append(f"""<h2>6. Global SVG references (R6)</h2>
<p>Number of global SVGs per reference, and the fraction of each list that the intersection reference of Kang et al. also contains.
Lists much larger than the intersection with high overlap indicate that the intersection discards genes other tests agree on.</p>{_table(agg)}""")
    return "\n".join(parts)


def section_uncertainty(out):
    summ = _load(out, "uncertainty_spots_summary.json", "json")
    mk = _load(out, "marker_check.csv")
    if summ is None:
        return ""
    parts = ["""<h2>7. Uncertainty: does it mean something biologically?</h2>
<p><b>Spots.</b> A 55 µm Visium spot holds roughly 1–10 cells. At a layer boundary it holds cells from both layers, so its
domain is genuinely ambiguous. If our per-spot uncertainty is biologically meaningful, confidence (agreement among kept methods)
and replicate stability (how often the spot stays with the same neighbours when the tissue is resampled) should be <em>lower at manually
annotated boundaries</em> (positive Spearman with distance to boundary) and confident spots should be more often correct.</p>"""]
    t = summ.set_index("sample").T.rename_axis(None, axis=1).reset_index().rename(columns={"index": "metric"})
    parts.append(_table(t))
    for d in sorted(p for p in out.iterdir() if p.is_dir()):
        f = d / "uncertainty_spots.csv"
        if f.exists():
            u = pd.read_csv(f)
            fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), layout="constrained")
            axes[0].scatter(u.x, -u.y, c=u.truth, s=3, cmap="tab10"); axes[0].set_title("manual layers")
            sc = axes[1].scatter(u.x, -u.y, c=u.confidence, s=3, cmap="viridis", vmin=0, vmax=1); axes[1].set_title("PCE confidence")
            plt.colorbar(sc, ax=axes[1], shrink=0.8)
            for a in axes[:2]:
                a.set_xticks([]); a.set_yticks([]); a.set_aspect("equal")
            bins = [0, 1.5, 3, 5, np.inf]; lab = ["≤1.5", "1.5–3", "3–5", ">5"]
            g = u.groupby(pd.cut(u.boundary_distance, bins, labels=lab), observed=False)
            axes[2].plot(lab, g.confidence.mean(), marker="o", label="confidence")
            if "replicate_stability" in u:
                axes[2].plot(lab, g.replicate_stability.mean(), marker="s", label="replicate stability")
            axes[2].plot(lab, g.correct.mean(), marker="^", ls="--", color="k", label="accuracy")
            axes[2].set_xlabel("distance to manual boundary (spots)"); axes[2].set_ylim(0, 1.02); axes[2].legend(fontsize=7, frameon=False)
            fig.suptitle(f"Slice {d.name}")
            parts.append(_img(fig))
            break
    if mk is not None and len(mk):
        parts.append("""<p><b>Genes.</b> Known DLPFC layer markers: their selection probability π and the layer they were assigned to.
π is the fraction of resampled replicates (new count split, new spot subsample, new method seeds) in which the gene was called
a domain-specific SVG. It measures <em>reproducibility under technical resampling</em>, not the probability that the gene is
biologically a marker. Expected layers come from the literature and are only a sanity check.</p>""" + _table(mk))
    return "\n".join(parts)


def section_maps(out):
    for d in sorted(p for p in out.iterdir() if p.is_dir()):
        f = d / "domain_labels.npz"
        if f.exists():
            z = np.load(f, allow_pickle=True)
            panels = [("manual layers", z["truth"], "tab10")]
            for key, title in [("ens__majority (Kang 2025)", "majority vote (Kang)"), ("ens__PCE (proposed)", "PCE (proposed)")]:
                if key in z.files:
                    panels.append((title, z[key], "tab10"))
            if "conf__PCE (proposed)" in z.files:
                panels.append(("PCE confidence", z["conf__PCE (proposed)"], "viridis"))
            fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.4))
            for ax, (t, v, cmap) in zip(np.atleast_1d(axes), panels):
                sc = ax.scatter(z["coords"][:, 0], -z["coords"][:, 1], c=v, s=3, cmap=cmap)
                ax.set_title(t); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
                if cmap == "viridis":
                    plt.colorbar(sc, ax=ax, fraction=0.046)
            fig.suptitle(f"Slice {d.name}")
            return f"<h2>Example slice</h2>{_img(fig)}"
    return ""


def build_report(out):
    out = Path(out)
    cfg = json.loads((out / "config.json").read_text()) if (out / "config.json").exists() else {}
    methods = json.loads((out / "methods.json").read_text()) if (out / "methods.json").exists() else {}
    errors = {d.name: json.loads((d / "error.json").read_text())["error"] for d in out.iterdir() if d.is_dir() and (d / "error.json").exists()}
    dom_html, _ = section_domains(out)
    body = f"""
<h1>Spatial SVG benchmark — real-data results</h1>
<p>Output folder: <code>{html.escape(str(out))}</code>. Samples: {', '.join(cfg.get('samples', []))}.
Methods run: {', '.join(methods.get('methods', []))}. SplitStab replicates B = {cfg.get('B')}.</p>
{('<p><b>Problems:</b> ' + html.escape('; '.join(methods.get('log', []) + [f'{k}: {v}' for k, v in errors.items()])) + '</p>') if (methods.get('log') or errors) else ''}
{section_maps(out)}
{dom_html}
{section_svgs(out)}
{section_uncertainty(out)}
<h2>How to read this report</h2>
<ul>
<li><b>PCE</b>: label-free method scores (stability × agreement × spatial coherence) → pre-registered pruning → weighted co-association consensus with per-spot confidence.</li>
<li><b>SplitStab combined (proposed)</b>: domains on one half of each count (count splitting), markers tested on the other half among confident spots with a depth-aware NB test (liberal per-replicate rule: q &lt; 0.1, rate ratio ≥ 1.2), a spatial gate (ACAT of Moran's I + SPARK-X-like on the held-out half), repeated B times with spot subsampling; synthetic NB negative-control genes calibrate the π threshold to an estimated FDP ≤ 0.1.</li>
<li><b>SplitStab strict (v2)</b>: the same with the strict per-replicate rule (q &lt; 0.05, rate ratio ≥ 1.5). <b>v1 stability</b>: the pilot method (perturbed replicates, liberal SpaGCN rule, same counts, π ≥ 0.6), kept as an ablation.</li>
<li><b>Kang 2025 pipeline</b>: one run per method → aligned majority vote → SpaGCN rule (q &lt; 0.05, in-domain fraction ≥ 0.8, fold change ≥ 1.5) on the same counts.</li>
<li>Gene-level outputs: <code>&lt;sample&gt;/domain_svgs.csv</code> (selection probability π per gene) and <code>global_svgs.csv</code>.</li>
</ul>"""
    style = """body{max-width:1050px;margin:2rem auto;padding:0 1rem;font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;line-height:1.5;color:#1f2328}
h1{border-bottom:1px solid #d0d7de}h2{border-bottom:1px solid #eee;margin-top:2rem}table.t{border-collapse:collapse;font-size:.85rem;margin:.6rem 0}
table.t th,table.t td{border:1px solid #d0d7de;padding:.25rem .5rem}table.t th{background:#f6f8fa}img{max-width:100%}code{background:#f6f8fa;padding:0 .2rem}"""
    path = out / "report.html"
    path.write_text(f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>SVG benchmark report</title><style>{style}</style></head><body>{body}</body></html>")
    return path


if __name__ == "__main__":
    print(build_report(sys.argv[1]))
