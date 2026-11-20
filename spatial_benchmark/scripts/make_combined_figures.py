"""Figures + summary for the combined-framework validation.
Usage: python scripts/make_combined_figures.py results/combined"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stbench.simulate import simulate  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
MAIN = ["Kang strict", "Kang liberal", "v1 stability (pi>=0.6)", "single split liberal", "SplitStab strict", "SplitStab liberal"]
LABEL = {"Kang strict": "Kang\nstrict", "Kang liberal": "Kang\nliberal", "v1 stability (pi>=0.6)": "v1\nstability",
         "single split liberal": "single split\n+ gate", "SplitStab strict": "SplitStab\nstrict (v2)", "SplitStab liberal": "Combined\n(proposed)"}
COL = {"Kang strict": "#969696", "Kang liberal": "#cccccc", "v1 stability (pi>=0.6)": "#9ecae1", "single split liberal": "#fdd0a2",
       "SplitStab strict": "#fdae6b", "SplitStab liberal": "#e6550d"}
ABL = ["Kang liberal", "v1 stability (pi>=0.6)", "single split liberal no gate", "single split liberal",
       "SplitStab liberal_nocore no gate", "SplitStab liberal no gate", "SplitStab liberal_nocore", "SplitStab liberal",
       "single split strict", "SplitStab strict"]
GLOB = ["global intersection (Kang)", "global ACAT full data (v1)", "global ACAT split + stability (combined)"]


def boot(x, n=4000):
    x = np.asarray(x, float); r = np.random.default_rng(0)
    return [round(float(v), 3) for v in np.quantile([r.choice(x, len(x)).mean() for _ in range(n)], [0.025, 0.975])]


def main(d):
    d = Path(d); f = d / "figures"; f.mkdir(exist_ok=True); S = {}
    df = pd.read_csv(d / "calls.csv", keep_default_na=False, na_values=["", "NaN", "nan"])  # "null" is a kind, not NA
    null = df[df.kind == "null"]; sig = df[df.kind == "signal"]
    # ---------------- gene level, main comparison
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), gridspec_kw={"width_ratios": [1, 1.4, 1.4]})
    m = null.groupby("approach").false_calls.agg(["mean", "min", "max"]).reindex(MAIN)
    axes[0].barh(range(len(MAIN)), m["mean"] + 0.3, color=[COL[a] for a in MAIN],
                 xerr=[m["mean"] - m["min"], m["max"] - m["mean"]], capsize=2)
    for i, (v, h) in enumerate(zip(m["mean"], m["max"])):
        axes[0].text(h + 2, i, f"{v:.1f}", va="center", fontsize=8)
    axes[0].set_yticks(range(len(MAIN))); axes[0].set_yticklabels([LABEL[a].replace("\n", " ") for a in MAIN]); axes[0].invert_yaxis()
    axes[0].set_xlabel("false calls on null tissues (no spatial genes)"); axes[0].set_title("A · tissues without signal", fontsize=9)
    for ax, depth, t in [(axes[1], 1.0, "B · full depth"), (axes[2], 0.25, "C · 25 % depth")]:
        s = sig[sig.depth == depth].groupby("approach")[["precision", "recall"]].mean().reindex(MAIN)
        x = np.arange(len(MAIN))
        ax.bar(x - .2, s.precision, .4, color=[COL[a] for a in MAIN], edgecolor="k", lw=.4, label="precision")
        ax.bar(x + .2, s.recall, .4, color=[COL[a] for a in MAIN], alpha=.5, hatch="///", edgecolor="k", lw=.4, label="recall")
        ax.set_xticks(x); ax.set_xticklabels([LABEL[a] for a in MAIN], fontsize=7); ax.set_ylim(0, 1.05)
        ax.set_title(f"{t} (6 tissues; solid = precision, hatched = recall)", fontsize=9)
    fig.tight_layout(); fig.savefig(f / "c1_gene_level.png", dpi=170); plt.close(fig)
    S["null_false_calls"] = null.groupby("approach").false_calls.agg(["mean", "min", "max"]).round(1).to_dict(orient="index")
    S["signal"] = {str(dep): g.groupby("approach")[["n_called", "false_calls", "precision", "recall", "f1"]].mean().round(3).to_dict(orient="index")
                   for dep, g in sig.groupby("depth")}
    # paired recall differences (combined vs others) per depth
    S["paired_recall_combined_minus"] = {}
    for dep, g in sig.groupby("depth"):
        p = g.pivot_table(index=["layout", "seed"], columns="approach", values="recall")
        S["paired_recall_combined_minus"][str(dep)] = {b: {"mean": round(float((p["SplitStab liberal"] - p[b]).mean()), 3),
                                                          "ci": boot(p["SplitStab liberal"] - p[b]),
                                                          "wins": int(((p["SplitStab liberal"] - p[b]) > 0).sum())}
                                                      for b in ["Kang strict", "SplitStab strict", "single split liberal", "v1 stability (pi>=0.6)"]}
    # ablation table
    abl = []
    for a in ABL:
        r = {"approach": a, "null_false_calls_mean": null[null.approach == a].false_calls.mean()}
        for dep in [1.0, 0.25]:
            g = sig[(sig.depth == dep) & (sig.approach == a)]
            r[f"precision_{dep}"] = g.precision.mean(); r[f"recall_{dep}"] = g.recall.mean(); r[f"false_calls_{dep}"] = g.false_calls.mean()
        abl.append(r)
    abl = pd.DataFrame(abl); abl.to_csv(d / "ablation.csv", index=False)
    S["ablation"] = abl.round(3).to_dict(orient="records")
    glob = []
    for a in GLOB:
        r = {"approach": a, "null_false_calls_mean": null[null.approach == a].false_calls.mean()}
        for dep in [1.0, 0.25]:
            g = sig[(sig.depth == dep) & (sig.approach == a)]
            r[f"precision_{dep}"] = g.precision.mean(); r[f"recall_{dep}"] = g.recall.mean()
        glob.append(r)
    S["global"] = pd.DataFrame(glob).round(3).to_dict(orient="records")

    # ---------------- uncertainty
    sp, gn, mt, pinc = [], [], [], []
    for z in sorted(d.glob("unc_*.npz")):
        tag = z.stem[4:]; layout, seed, depth = tag.split("_"); u = np.load(z, allow_pickle=True)
        s = pd.DataFrame({k[5:]: u[k] for k in u.files if k.startswith("spot_")}); s["tag"] = tag; s["depth"] = float(depth); sp.append(s)
        sim = simulate(layout=layout, seed=int(seed))
        g = pd.DataFrame({k[5:]: u[k] for k in u.files if k.startswith("gene_") and k != "gene_pi_nc"}); g["tag"] = tag; g["depth"] = float(depth)
        pinc.append(u["gene_pi_nc"])
        g["mean_expr"] = sim.counts.mean(0); gn.append(g)
        mf = d / f"methods_{tag}.csv"
        if mf.exists():
            m_ = pd.read_csv(mf); m_["tag"] = tag; m_["depth"] = float(depth); mt.append(m_)
    sp = pd.concat(sp); gn = pd.concat(gn); mt = pd.concat(mt) if mt else None
    full = sp[sp.depth == 1.0].copy()
    bins = [0, 1.01, 2.01, 3.01, np.inf]; blab = ["1", "2", "3", "≥4"]
    full["bd"] = pd.cut(full.boundary_dist, bins, labels=blab)
    full["mix"] = pd.cut(full.membership, [0, .6, .8, .95, 1.01], labels=["<0.6", "0.6–0.8", "0.8–0.95", ">0.95"])
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.3))
    for col, lab, mk in [("confidence", "PCE confidence", "o"), ("stability", "replicate stability", "s"), ("correct", "accuracy", "^")]:
        g = full.groupby("bd", observed=False)[col].mean(); axes[0].plot(blab, g.values, marker=mk, label=lab, ls="--" if col == "correct" else "-")
        g = full.groupby("mix", observed=False)[col].mean(); axes[1].plot(g.index.astype(str), g.values, marker=mk, label=lab, ls="--" if col == "correct" else "-")
    axes[0].set_xlabel("distance to true domain boundary (spots)"); axes[0].set_ylim(0, 1.02); axes[0].legend(fontsize=7, frameon=False)
    axes[0].set_title("A · uncertainty sits at boundaries", fontsize=9)
    axes[1].set_xlabel("true share of the spot's main domain (mixing)"); axes[1].set_ylim(0, 1.02)
    axes[1].set_title("B · uncertainty tracks true cell mixing", fontsize=9)
    edges = np.linspace(0, 1, 11)
    for col, mk in [("confidence", "o"), ("stability", "s"), ("agreement", "x")]:
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            s_ = (full[col] >= lo) & (full[col] < hi if hi < 1 else full[col] <= hi)
            if s_.sum() >= 30:
                xs.append(full[col][s_].mean()); ys.append(full.correct[s_].mean())
        axes[2].plot(xs, ys, marker=mk, label={"agreement": "label agreement (aligned)", "stability": "replicate stability"}.get(col, col),
                     ls=":" if col == "agreement" else "-")
    axes[2].plot([0, 1], [0, 1], ls=":", color="k", lw=.8); axes[2].set_xlabel("stated certainty"); axes[2].set_ylabel("observed accuracy")
    axes[2].legend(fontsize=7, frameon=False); axes[2].set_title("C · reliability (diagonal = calibrated)", fontsize=9)
    fig.tight_layout(); fig.savefig(f / "c2_spot_uncertainty.png", dpi=170); plt.close(fig)
    S["spot"] = {}
    for col in ["confidence", "stability", "agreement"]:
        ok = np.isfinite(full[col])
        S["spot"][col] = {"spearman_vs_boundary_dist": round(float(spearmanr(full[col][ok], full.boundary_dist[ok])[0]), 3),
                          "spearman_vs_membership": round(float(spearmanr(full[col][ok], full.membership[ok])[0]), 3),
                          "by_boundary_dist": full.groupby("bd", observed=False)[col].mean().round(3).to_dict(),
                          "acc_ge_0.6": round(float(full.correct[full[col] >= 0.6].mean()), 3),
                          "acc_lt_0.6": round(float(full.correct[full[col] < 0.6].mean()), 3),
                          "share_ge_0.6": round(float((full[col] >= 0.6).mean()), 3)}
    S["spot"]["accuracy_by_boundary_dist"] = full.groupby("bd", observed=False).correct.mean().round(3).to_dict()
    S["spot"]["accuracy_by_mixing"] = full.groupby("mix", observed=False).correct.mean().round(3).to_dict()
    # genes
    g1 = gn[gn.depth == 1.0].copy()
    dom = g1[g1.gene_class == "domain"].copy()
    dom["absfc"] = np.abs(dom.logfc)
    dom["fc_bin"] = pd.qcut(dom.absfc, 4, duplicates="drop")
    dom["ex_bin"] = pd.qcut(dom.mean_expr, 4, duplicates="drop")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.3))
    for col, lab in [("pi", "combined"), ("pi_strict", "strict (v2)"), ("pi_v1", "v1 (same data)")]:
        g = dom.groupby("fc_bin", observed=False)[col].mean()
        axes[0].plot([f"{iv.left:.2f}–{iv.right:.2f}" for iv in g.index], g.values, marker="o", label=lab)
    axes[0].set_xlabel("true |log fold change| of domain marker (quartiles)"); axes[0].set_ylabel("mean π"); axes[0].set_ylim(0, 1.02)
    axes[0].legend(fontsize=7, frameon=False); axes[0].set_title("A · π rises with effect size", fontsize=9)
    pbins = [0, 0.001, 0.25, 0.5, 0.75, 1.01]; plab = ["0", "0–0.25", "0.25–0.5", "0.5–0.75", "≥0.75"]
    w = 0.27
    for j, (cls, c) in enumerate([("domain", "#e6550d"), ("gradient", "#3182bd"), ("null", "#969696")]):
        v = g1[g1.gene_class == cls].pi
        share = pd.cut(v, pbins, labels=plab, right=False).value_counts(normalize=True).reindex(plab)
        axes[1].bar(np.arange(len(plab)) + (j - 1) * w, share.values, w, color=c, label=f"{cls} genes (n={len(v)})")
    axes[1].set_xticks(range(len(plab))); axes[1].set_xticklabels(plab); axes[1].set_xlabel("π (combined)"); axes[1].set_ylabel("share of genes")
    axes[1].legend(fontsize=7, frameon=False); axes[1].set_title("B · π separates gene classes", fontsize=9)
    sel = dom[dom.pi >= 0.5]
    corr = (sel.modal_true == sel.target)
    pb = pd.cut(sel.purity, [0, .6, .8, .95, 1.01])
    gg = corr.groupby(pb, observed=False).mean()
    axes[2].bar([str(i) for i in gg.index], gg.values, color="#e6550d")
    axes[2].set_xlabel("domain purity (share of replicates agreeing on the domain)"); axes[2].set_ylabel("assigned to the right domain")
    axes[2].set_ylim(0, 1.02); axes[2].set_title("C · domain assignment of selected markers", fontsize=9)
    fig.tight_layout(); fig.savefig(f / "c3_gene_uncertainty.png", dpi=170); plt.close(fig)
    S["gene"] = {"pi_by_fc_quartile": {str(k): round(float(v), 3) for k, v in dom.groupby("fc_bin", observed=False).pi.mean().items()},
                 "pi_by_expr_quartile": {str(k): round(float(v), 3) for k, v in dom.groupby("ex_bin", observed=False).pi.mean().items()},
                 "mean_pi_by_class": g1.groupby("gene_class").pi.mean().round(3).to_dict(),
                 "share_pi_ge_0.5_by_class": g1.groupby("gene_class").pi.apply(lambda v: float((v >= .5).mean())).round(3).to_dict(),
                 "modal_domain_correct_selected_domain_genes": round(float(corr.mean()), 3),
                 "modal_domain_correct_by_purity": {str(k): round(float(v), 3) for k, v in gg.items()},
                 "n_selected_domain_genes": int(len(sel)),
                 "mean_purity_selected": g1[g1.pi >= .5].groupby("gene_class").purity.mean().round(3).to_dict(),
                 "pi_nc_max": round(float(max(v.max() for v in pinc)), 3), "pi_nc_mean": round(float(np.mean([v.mean() for v in pinc])), 4)}
    # pi calibration (fraction true) by pi bin, combined
    pb2 = pd.cut(g1.pi, [0.001, 0.25, 0.5, 0.75, 0.95, 1.0])
    S["gene"]["truth_rate_by_pi"] = {str(k): [round(float(v), 3), int(n)] for (k, v), n in
                                     zip(g1.groupby(pb2, observed=False).truth.mean().items(), g1.groupby(pb2, observed=False).size())}
    # method level
    if mt is not None:
        m1 = mt[mt.depth == 1.0]
        best = m1.loc[m1.groupby(["tag", "b"]).ari.idxmax()]
        pbest = best.groupby("tag").method.value_counts(normalize=True).unstack(fill_value=0)
        kept = m1.groupby(["tag", "method"]).kept.mean().unstack()
        S["methods"] = {"P_best_by_tissue": pbest.round(2).to_dict(orient="index"), "keep_rate_by_tissue": kept.round(2).to_dict(orient="index"),
                        "P_best_overall": best.method.value_counts(normalize=True).round(3).to_dict(),
                        "keep_rate_overall": m1.groupby("method").kept.mean().round(3).to_dict(),
                        "mean_ari": m1.groupby("method").ari.mean().round(3).to_dict(),
                        "share_replicates_best_kept": round(float(best.kept.mean()), 3)}
        fig, ax = plt.subplots(figsize=(6.5, 3))
        ms = sorted(m1.method.unique()); x = np.arange(len(ms))
        ax.bar(x - .2, [S["methods"]["P_best_overall"].get(m, 0) for m in ms], .4, label="P(best) across replicates (needs truth)", color="#3182bd")
        ax.bar(x + .2, [S["methods"]["keep_rate_overall"].get(m, 0) for m in ms], .4, label="kept by label-free pruning", color="#e6550d")
        ax.set_xticks(x); ax.set_xticklabels(ms, fontsize=8); ax.set_ylim(0, 1.05); ax.legend(fontsize=7, frameon=False)
        ax.set_title("Method-level uncertainty: which method is best, and does pruning keep it?", fontsize=9)
        fig.tight_layout(); fig.savefig(f / "c4_method_uncertainty.png", dpi=170); plt.close(fig)
    info = json.loads((d / "info.json").read_text())
    S["thresholds"] = [i["thresholds"] for i in info]
    S["pce_ari"] = {f"{i['layout']}_{i['seed']}_{i['depth']}": i.get("pce_ari") for i in info if i.get("pce_ari") is not None}
    (d / "summary.json").write_text(json.dumps(S, indent=1, default=str))
    print(json.dumps(S, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/combined")
