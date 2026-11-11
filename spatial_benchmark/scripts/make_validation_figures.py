"""Figures + summary for the held-out validation. Usage: python scripts/make_validation_figures.py results/validation"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
SINGLE = ["kmeans_pca", "gmm_pca", "ward_spatial", "spectral_spatial", "kmeans_smooth"]
ENS = ["majority (Kang 2025)", "STCC-average", "EnSDD-weights", "CAC weighted", "PCE (proposed)"]


def boot(x, n=4000):
    x = np.asarray(x, float); r = np.random.default_rng(0)
    return np.quantile([r.choice(x, len(x)).mean() for _ in range(n)], [0.025, 0.975]).tolist()


def main(d):
    d = Path(d); f = d / "figures"; f.mkdir(exist_ok=True); S = {}
    # V1
    v1 = pd.read_csv(d / "v1_domains.csv")
    p = v1.pivot_table(index=["layout", "seed"], columns="approach", values="ari")
    order = SINGLE + ENS
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for i, a in enumerate(order):
        v = p[a].values
        col = "#9ecae1" if a in SINGLE else ("#bdbdbd" if "majority" in a else ("#e6550d" if "PCE" in a else "#fdae6b"))
        ax.bar(i, v.mean(), color=col, width=0.7); ax.scatter(np.full(len(v), i) + np.linspace(-.2, .2, len(v)), v, s=9, color="k", zorder=3)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=30, ha="right"); ax.set_ylabel("ARI to true domains")
    ax.set_title("V1 · domain aggregation on held-out tissues (dots = 6 tissues)")
    fig.tight_layout(); fig.savefig(f / "v1_domains.png", dpi=170); plt.close(fig)
    S["v1_mean"] = p[order].mean().round(3).to_dict()
    for b in ["majority (Kang 2025)", "STCC-average", "EnSDD-weights", "CAC weighted"]:
        diff = p["PCE (proposed)"] - p[b]
        S[f"v1_pce_minus_{b}"] = {"mean": round(float(diff.mean()), 3), "wins": int((diff > 0).sum()), "ci": [round(x, 3) for x in boot(diff)]}
    S["v1_pce_minus_best_single"] = round(float((p["PCE (proposed)"] - p[SINGLE].max(axis=1)).mean()), 3)
    S["v1_calibration"] = v1[v1.kind == "ensemble"].groupby("approach")[["ece", "acc_conf_ge_0.6", "acc_conf_lt_0.6", "share_core"]].mean().round(3).dropna(how="all").to_dict(orient="index")
    # V2
    v2 = pd.read_csv(d / "v2_double_dipping.csv")
    cols = ["paper rule, same data (liberal)", "v2 rule, same data", "paper rule, same data (strict)", "Wilcoxon rule, count split",
            "v2 rule, count split", "SplitStab without spatial gate", "v2 rule, count split + spatial gate", "SplitStab (selected)"]
    cols = [c for c in cols if c in v2]
    fig, ax = plt.subplots(figsize=(7.5, 3.3))
    m = v2[cols].mean(); lo = v2[cols].min(); hi = v2[cols].max()
    colors = ["#bdbdbd" if "paper" in c else ("#e6550d" if ("gate" in c and "without" not in c) or c == "SplitStab (selected)" else "#fdae6b") for c in cols]
    ax.barh(range(len(cols)), m.values + 0.3, color=colors, xerr=[m - lo, hi - m], capsize=2)
    for i, (v, h) in enumerate(zip(m.values, hi.values)):
        ax.text(h + 2, i, f"{v:.1f}", va="center", fontsize=8)
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(["Kang rule, liberal, same data", "v2 test, same data", "Kang rule (strict), same data",
                                                         "Wilcoxon rule, count split", "v2 test, count split", "SplitStab without spatial gate",
                                                         "v2 test, count split + spatial gate", "SplitStab (proposed)"][:len(cols)])
    ax.invert_yaxis(); ax.set_xlabel("domain-SVG calls on tissues with NO spatial signal (all false)")
    ax.set_title("V2 · false domain-SVG calls on 4 null tissues (bar = mean, whiskers = range)", fontsize=9)
    fig.tight_layout(); fig.savefig(f / "v2_null.png", dpi=170); plt.close(fig)
    S["v2_null_calls_mean"] = m.round(1).to_dict(); S["v2_null_calls_range"] = {c: [int(lo[c]), int(hi[c])] for c in cols}
    # V3 / V5
    v3 = pd.read_csv(d / "v3_splitstab.csv")
    S["v3"] = v3.groupby("depth").mean(numeric_only=True).round(3).to_dict(orient="index")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    full = v3[v3.depth == 1.0]
    names = [("paper_strict", "Kang pipeline (strict)", "#969696"), ("paper_liberal", "Kang pipeline (liberal)", "#cccccc"), ("splitstab", "SplitStab (proposed)", "#e6550d")]
    for i, (k, lab, c) in enumerate(names):
        for j, met in enumerate(["precision", "recall"]):
            v = full[f"{k}_{met}"].values
            axes[0].bar(i * 3 + j, np.nanmean(v), color=c, alpha=1 if met == "precision" else 0.6)
            axes[0].scatter(np.full(len(v), i * 3 + j), v, s=7, color="k", zorder=3)
    axes[0].set_xticks([0.5, 3.5, 6.5]); axes[0].set_xticklabels(["Kang strict", "Kang liberal", "SplitStab"])
    axes[0].set_ylim(0, 1.05); axes[0].set_title("V3 · precision (dark) and recall (light), full depth", fontsize=9)
    # pi calibration
    pis = json.loads((d / "v3_pi.json").read_text())
    allpi = np.concatenate([np.array(v["pi"]) for k, v in pis.items() if k.endswith("_1.0")])
    allt = np.concatenate([np.array(v["truth"]) for k, v in pis.items() if k.endswith("_1.0")])
    bins = [0.01, 0.2, 0.4, 0.6, 0.8, 0.95, 1.01]; xs, ys = [], []
    for a, b in zip(bins[:-1], bins[1:]):
        s = (allpi >= a) & (allpi < b)
        if s.sum() >= 5:
            xs.append(allpi[s].mean()); ys.append(allt[s].mean())
    axes[1].plot(xs, ys, marker="o", color="#e6550d"); axes[1].plot([0, 1], [0, 1], ls=":", color="k", lw=.8)
    axes[1].set_xlabel("selection probability π"); axes[1].set_ylabel("fraction truly SVG"); axes[1].set_title("V3 · selected genes by π: almost all true (π is conservative)", fontsize=9)
    S["v3_pi_calibration"] = [[round(a, 3), round(b, 3)] for a, b in zip(xs, ys)]
    # depth
    g = v3.groupby("depth")
    for k, lab, c in [names[0], names[2]]:
        mm = g[f"{k}_recall"].mean(); pp = g[f"{k}_precision"].mean()
        axes[2].plot(mm.index, mm.values, marker="o", color=c, label=f"{lab}: recall")
        axes[2].plot(pp.index, pp.values, marker="s", ls="--", color=c, label=f"{lab}: precision")
    axes[2].set_xscale("log"); axes[2].invert_xaxis(); axes[2].set_xlabel("sequencing depth (fraction kept)"); axes[2].set_ylim(0, 1.05)
    axes[2].legend(fontsize=6, frameon=False); axes[2].set_title("V5 · low depth", fontsize=9)
    fig.tight_layout(); fig.savefig(f / "v3_v5_splitstab.png", dpi=170); plt.close(fig)
    # V4
    v4 = pd.read_csv(d / "v4_global.csv"); v4 = v4[~v4.reference.str.startswith("LOO")]
    m4 = v4.groupby("reference")[["precision", "recall", "f1", "n_called"]].mean()
    order4 = ["intersection (Kang 2025)", "Castl-like rank_agg", "Castl-like NC-FDP", "Castl-like Cauchy + BY", "ACAT + BH (ours)"]
    m4 = m4.loc[order4]
    fig, ax = plt.subplots(figsize=(6.5, 3)); x = np.arange(len(m4))
    ax.bar(x - .2, m4.precision, .4, label="precision", color="#3182bd"); ax.bar(x + .2, m4.recall, .4, label="recall", color="#e6550d")
    ax.set_xticks(x); ax.set_xticklabels([s.replace(" (", "\n(").replace("Castl-like ", "Castl-like\n") for s in m4.index], fontsize=7)
    ax.set_ylim(0, 1.05); ax.legend(fontsize=7, frameon=False); ax.set_title("V4 · global SVG references vs truth (6 held-out tissues)", fontsize=9)
    fig.tight_layout(); fig.savefig(f / "v4_global.png", dpi=170); plt.close(fig)
    S["v4"] = m4.round(3).to_dict(orient="index")
    (d / "summary.json").write_text(json.dumps(S, indent=1))
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/validation")
