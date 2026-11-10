"""Figures and summary tables for the pilot. Usage: python scripts/make_figures.py --pilot results/pilot"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

SINGLE = ["kmeans_pca", "gmm_pca", "ward_spatial", "spectral_spatial", "kmeans_smooth"]
ENS = ["majority", "cac", "cac_weighted", "cac_weighted_pruned"]
LABEL = {"kmeans_pca": "k-means (PCA)", "gmm_pca": "GMM (PCA)", "ward_spatial": "Ward + spatial",
         "spectral_spatial": "Spectral expr+space", "kmeans_smooth": "k-means (smoothed)",
         "majority": "Majority vote (paper)", "cac": "Co-association", "cac_weighted": "Co-assoc. weighted",
         "cac_weighted_pruned": "Co-assoc. weighted + pruned"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def boot_ci(x, n=4000, seed=0):
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    b = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return np.quantile(b, [0.025, 0.975])


def f1_aggregation(d, out):
    e1 = pd.read_csv(d / "e1.csv")
    single = e1[e1.kind == "single"].groupby(["layout", "seed", "approach"]).ari.mean().reset_index()
    ens = e1[e1.kind == "ensemble"][["layout", "seed", "approach", "ari"]]
    df = pd.concat([single, ens])
    order = SINGLE + ENS
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for i, a in enumerate(order):
        v = df[df.approach == a].ari.values
        col = "#9ecae1" if a in SINGLE else ("#bdbdbd" if a == "majority" else "#e6550d")
        ax.bar(i, v.mean(), color=col, width=0.7)
        ax.scatter(np.full(len(v), i) + np.linspace(-0.2, 0.2, len(v)), v, s=9, color="k", zorder=3)
    ax.set_xticks(range(len(order))); ax.set_xticklabels([LABEL[a] for a in order], rotation=30, ha="right")
    ax.set_ylabel("ARI to true domains"); ax.set_title("Aggregating domain callers (6 simulated tissues; dots = tissues)")
    fig.tight_layout(); fig.savefig(out / "f1_domain_aggregation.png", dpi=170); plt.close(fig)
    piv = df.pivot_table(index=["layout", "seed"], columns="approach", values="ari")
    return piv


def f2_pruning(d, out):
    p = pd.read_csv(d / "prune.csv")
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for order, col in [("label_free", "#e6550d"), ("oracle", "#636363")]:
        g = p[p.order == order].groupby("q").ari.agg(["mean", "std"])
        ax.errorbar(g.index, g["mean"], yerr=g["std"], marker="o", color=col, capsize=2,
                    label="ranked by label-free score" if order == "label_free" else "ranked by true ARI (oracle)")
    ax.set_xlabel("methods kept (top-q)"); ax.set_ylabel("ensemble ARI"); ax.set_xticks(range(1, 6))
    ax.legend(fontsize=7, frameon=False); ax.set_title("Ensemble pruning curve")
    fig.tight_layout(); fig.savefig(out / "f2_pruning_curve.png", dpi=170); plt.close(fig)
    return p.groupby(["order", "q"]).ari.mean().unstack(0)


def f3_calibration(d, out):
    c = pd.read_csv(d / "calib.csv")
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    for mode, col in [("majority", "#969696"), ("cac_weighted_pruned", "#e6550d")]:
        g = c[c["mode"] == mode].groupby("bin").apply(lambda x: pd.Series({"conf": np.average(x.mean_conf, weights=x.n),
                                                                          "acc": np.average(x.accuracy, weights=x.n), "n": x.n.sum()}))
        ax.plot(g.conf, g.acc, marker="o", color=col, label=LABEL[mode])
    ax.plot([0, 1], [0, 1], ls=":", color="k", lw=0.8)
    ax.set_xlabel("per-spot confidence"); ax.set_ylabel("fraction of spots correctly assigned"); ax.legend(fontsize=7, frameon=False)
    ax.set_title("Is spot-level confidence meaningful?")
    fig.tight_layout(); fig.savefig(out / "f3_confidence_calibration.png", dpi=170); plt.close(fig)
    rows = []
    for mode in ["majority", "cac_weighted_pruned"]:
        g = c[c["mode"] == mode]
        hi = g[g.mid >= 0.75]; lo = g[g.mid < 0.75]
        rows.append({"mode": mode, "acc_conf>=0.7": np.average(hi.accuracy, weights=hi.n) if len(hi) else np.nan,
                     "acc_conf<0.7": np.average(lo.accuracy, weights=lo.n) if len(lo) else np.nan,
                     "share_conf>=0.7": hi.n.sum() / g.n.sum()})
    return pd.DataFrame(rows)


def f4_reference(d, out):
    e2 = pd.read_csv(d / "e2.csv"); e3 = pd.read_csv(d / "e3.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))
    g = e3[e3.reference.isin(["morans_i", "gearys_c", "sparkx_like", "intersection", "intersection_rescued", "acat"])]
    m = g.groupby("reference")[["precision", "recall"]].mean().loc[["morans_i", "gearys_c", "sparkx_like", "intersection", "intersection_rescued", "acat"]]
    x = np.arange(len(m))
    axes[0].bar(x - 0.18, m.precision, 0.36, label="precision", color="#3182bd")
    axes[0].bar(x + 0.18, m.recall, 0.36, label="recall", color="#e6550d")
    axes[0].set_xticks(x); axes[0].set_xticklabels(["Moran's I", "Geary's C", "SPARK-X-like", "Intersection\n(paper)", "Intersection\n+ rescue", "ACAT\n(proposed)"], fontsize=7, rotation=25, ha="right")
    axes[0].set_ylim(0, 1.05); axes[0].legend(fontsize=7, frameon=False); axes[0].set_title("Global SVG reference vs truth")
    s = e2[e2.reference.isin(["truth (all SVGs)", "intersection consensus"])].pivot_table(index=["layout", "seed", "partition"], columns="reference", values="recall").reset_index()
    for part, grp in s.groupby("partition"):
        col = "#9ecae1" if part in SINGLE else ("#969696" if part == "majority" else "#e6550d")
        axes[1].scatter(grp["truth (all SVGs)"], grp["intersection consensus"], s=14, color=col, label=LABEL.get(part, part))
    lim = [0, max(0.6, s[["truth (all SVGs)", "intersection consensus"]].max().max() + 0.05)]
    axes[1].plot(lim, lim, ls=":", color="k", lw=0.8)
    axes[1].set_xlabel("recall against true SVGs"); axes[1].set_ylabel("recall against consensus reference")
    axes[1].set_title("Domain-SVG sensitivity: true vs paper's reference", fontsize=9)
    h, l = axes[1].get_legend_handles_labels(); uniq = dict(zip(l, h))
    axes[1].legend(uniq.values(), uniq.keys(), fontsize=6, frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(out / "f4_reference_bias.png", dpi=170); plt.close(fig)
    # does the reference change the method ranking? Kendall tau per tissue
    taus = []
    for (l, sd), grp in s.groupby(["layout", "seed"]):
        taus.append(stats.kendalltau(grp["truth (all SVGs)"], grp["intersection consensus"]).statistic)
    precs = e2[e2.reference.isin(["truth (all SVGs)", "intersection consensus"])].groupby("reference")[["precision", "recall"]].mean()
    return m, float(np.nanmean(taus)), precs, e3


def f5_robustness(d, out):
    r = pd.read_csv(d / "e4_robustness.csv")
    axes_ = [("depth", "sequencing depth (fraction kept)"), ("spot_frac", "spots kept (fraction)"), ("coord_sd", "coordinate jitter (spot spacings)")]
    approaches = [("kmeans_pca", "strict"), ("majority", "strict"), ("cac_weighted_pruned", "strict"), ("cac_weighted_pruned", "liberal"),
                  ("global:intersection_rescued", "global"), ("global:acat", "global")]
    colors = {"kmeans_pca": "#9ecae1", "majority": "#969696", "cac_weighted_pruned": "#e6550d", "global:intersection_rescued": "#756bb1", "global:acat": "#31a354"}
    fig, axs = plt.subplots(3, 3, figsize=(10, 7.8))
    for j, (ax_name, xl) in enumerate(axes_):
        sub = r[r.axis == ax_name]
        for i, metric in enumerate(["ari", "svg_f1", "jaccard_to_unperturbed"]):
            ax = axs[i, j]
            for a, rule in approaches:
                g = sub[(sub.approach == a) & (sub.rule == rule)].groupby("level")[metric].agg(["mean", "std"])
                if metric == "ari" and a.startswith("global"):
                    continue
                if g["mean"].isna().all():
                    continue
                ax.errorbar(g.index, g["mean"], yerr=g["std"], marker="o", ms=3, capsize=2, color=colors[a],
                            ls="-" if rule != "liberal" else "--", label={"global:intersection_rescued": "Global reference: intersection (paper)",
                                                                           "global:acat": "Global reference: ACAT (proposed)"}.get(a, f"{LABEL.get(a, a)} — {rule} SVG rule"))
            if ax_name == "depth":
                ax.set_xscale("log"); ax.invert_xaxis()
            if ax_name == "spot_frac":
                ax.invert_xaxis()
            ax.set_xlabel(xl if i == 2 else ""); ax.set_ylabel({"ari": "domain ARI", "svg_f1": "SVG F1 vs truth", "jaccard_to_unperturbed": "SVG-set Jaccard\nvs unperturbed"}[metric] if j == 0 else "")
    h, l = axs[2, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7, frameon=False)
    fig.suptitle("Robustness: one-factor-at-a-time perturbations (mean ± SD over 6 tissues × 3 replicates)", fontsize=10)
    fig.tight_layout(rect=[0, 0.07, 1, 0.97]); fig.savefig(out / "f5_robustness.png", dpi=170); plt.close(fig)
    # robustness AUC: mean metric over the non-baseline levels divided by baseline
    rows = []
    for (ax_name, a, rule), g in r.groupby(["axis", "approach", "rule"]):
        base = g[g.level == {"depth": 1.0, "spot_frac": 1.0, "coord_sd": 0.0}[ax_name]].svg_f1.mean()
        pert = g[g.level != {"depth": 1.0, "spot_frac": 1.0, "coord_sd": 0.0}[ax_name]].svg_f1.mean()
        rows.append({"axis": ax_name, "approach": a, "rule": rule, "f1_baseline": base, "f1_perturbed_mean": pert,
                     "retention": pert / base if base > 0 else np.nan,
                     "jaccard_mean": g[g.level != {"depth": 1.0, "spot_frac": 1.0, "coord_sd": 0.0}[ax_name]].jaccard_to_unperturbed.mean()})
    return pd.DataFrame(rows)


def f6_uq(d, out):
    ss = pd.read_csv(d / "e5_stability_selection.csv")
    truth = json.loads((d / "truth_svg.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    # (a) calibration of selection probability (liberal rule, proposed pipeline)
    for mode, col in [("majority_liberal", "#969696"), ("cac_weighted_pruned_liberal", "#e6550d")]:
        pis, ts = [], []
        for key, t in truth.items():
            f = d / f"pi_{key}_{mode}.npy"
            if f.exists():
                pis.append(np.load(f)); ts.append(np.array(t))
        pi = np.concatenate(pis); t = np.concatenate(ts)
        bins = [0.001, 0.2, 0.4, 0.6, 0.8, 0.999, 1.001]
        xs, ys, ns = [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            sel = (pi >= lo) & (pi < hi)
            if sel.sum() >= 5:
                xs.append(pi[sel].mean()); ys.append(t[sel].mean()); ns.append(sel.sum())
        axes[0].plot(xs, ys, marker="o", color=col, label=mode.replace("_liberal", "").replace("cac_weighted_pruned", "proposed").replace("majority", "majority vote"))
    axes[0].set_xlabel("selection probability π (20 replicates)"); axes[0].set_ylabel("fraction truly spatially variable")
    axes[0].set_title("Selection probability is informative\n(liberal SpaGCN rule)", fontsize=9); axes[0].legend(fontsize=7, frameon=False)
    # (b) stability selection vs single run: precision and recall
    g = ss[(ss["mode"] == "cac_weighted_pruned|liberal") & (ss.pi_thr >= 0.6)].groupby("pi_thr")[["precision", "recall", "observed_FP", "mb_bound_EV", "single_run_mean_FP"]].mean()
    r4 = pd.read_csv(d / "e4_robustness.csv")
    b = r4[(r4.axis == "depth") & (r4.level == 1.0) & (r4.approach == "cac_weighted_pruned")].groupby("rule")[["svg_precision", "svg_recall"]].mean()
    axes[1].plot(g.index, g.precision, marker="o", color="#3182bd", label="precision, stability-selected (liberal rule)")
    axes[1].plot(g.index, g.recall, marker="o", color="#e6550d", label="recall, stability-selected (liberal rule)")
    axes[1].axhline(b.loc["liberal", "svg_precision"], color="#3182bd", ls="--", lw=0.8, label="precision, single run (liberal)")
    axes[1].axhline(b.loc["strict", "svg_recall"], color="#e6550d", ls=":", lw=1, label="recall, single run (paper's strict rule)")
    axes[1].set_xlabel("π threshold"); axes[1].set_ylim(0, 1.02); axes[1].legend(fontsize=6, frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.45))
    axes[1].set_title("Stability selection trade-off", fontsize=9)
    # (c) error control: observed false positives vs MB bound and vs a single run
    axes[2].plot(g.index, g.observed_FP, marker="o", color="#e6550d", label="observed FP (stability-selected)")
    axes[2].plot(g.index, g.mb_bound_EV, marker="s", color="k", ls=":", label="MB bound on E[FP]")
    axes[2].axhline(g.single_run_mean_FP.iloc[0], color="#969696", ls="--", label="mean FP of a single run")
    axes[2].set_yscale("log"); axes[2].set_ylim(0.1, 1000); axes[2].set_xlabel("π threshold"); axes[2].legend(fontsize=7, frameon=False)
    axes[2].set_title("False positives under control", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "f6_uncertainty.png", dpi=170); plt.close(fig)
    return ss.groupby(["mode", "pi_thr"])[["n_called", "precision", "recall", "observed_FP", "mb_bound_EV", "single_run_mean_FP"]].mean()


def f7_rank_uncertainty(d, out):
    r = pd.read_csv(d / "e5_rank_uncertainty.csv")
    p = r.best.value_counts(normalize=True).reindex(SINGLE).fillna(0)
    per = r.groupby(["layout", "seed"]).best.agg(lambda x: x.value_counts(normalize=True).iloc[0])
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax.barh([LABEL[m] for m in p.index], p.values, color="#9ecae1")
    ax.set_xlabel("P(method ranks first) over 120 bootstrap replicates"); ax.set_title("How certain is the benchmark's winner?", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "f7_rank_uncertainty.png", dpi=170); plt.close(fig)
    return p, per


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pilot", default="results/pilot"); args = ap.parse_args()
    d = Path(args.pilot); out = d / "figures"; out.mkdir(exist_ok=True)
    summary = {}
    piv = f1_aggregation(d, out)
    summary["e1_mean_ari"] = piv.mean().round(3).to_dict()
    summary["e1_ci"] = {k: [round(x, 3) for x in boot_ci(piv[k])] for k in piv.columns}
    summary["e1_cacwp_minus_majority"] = {"mean": round(float((piv.cac_weighted_pruned - piv.majority).mean()), 3),
                                          "wins": int((piv.cac_weighted_pruned > piv.majority).sum()), "n": int(len(piv)),
                                          "ci": [round(x, 3) for x in boot_ci(piv.cac_weighted_pruned - piv.majority)]}
    summary["e1_cacwp_vs_best_single"] = round(float((piv.cac_weighted_pruned - piv[SINGLE].max(1)).mean()), 3)
    summary["e1_cacwp_vs_mean_single"] = round(float((piv.cac_weighted_pruned - piv[SINGLE].mean(1)).mean()), 3)
    summary["e2_pruning_curve"] = f2_pruning(d, out).round(3).to_dict()
    summary["e3_calibration"] = f3_calibration(d, out).round(3).to_dict(orient="records")
    m, tau, precs, e3 = f4_reference(d, out)
    summary["e4_reference_quality"] = m.round(3).to_dict(orient="index")
    summary["e4_kendall_tau_truth_vs_consensus_ranking"] = round(tau, 3)
    summary["e4_domain_svg_pr_by_reference"] = precs.round(3).to_dict(orient="index")
    summary["e4_aupr"] = e3.groupby("reference").aupr.mean().dropna().round(3).to_dict()
    summary["e4_null_fpr"] = e3.groupby("reference")["null_fpr_at_0.05"].mean().dropna().round(4).to_dict()
    summary["e4_agreement_single_methods"] = round(float(pd.read_csv(d / "agreement.csv").agreement_single_methods.mean()), 3)
    summary["e5_robustness"] = f5_robustness(d, out).round(3).to_dict(orient="records")
    summary["e6_stability_selection"] = f6_uq(d, out).round(3).reset_index().to_dict(orient="records")
    p, per = f7_rank_uncertainty(d, out)  # figure kept out of PROPOSAL.md: P(best) = 1 in the pilot
    summary["e7_p_best"] = p.round(3).to_dict(); summary["e7_p_modal_best_per_tissue"] = {f"{k[0]}_{k[1]}": v for k, v in per.round(3).to_dict().items()}
    (d / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
