"""Period-alignment ablation: does giving the Fourier basis the biological period help?

Compares each model on circadian_liver / circadian_blood with the default time
scaling (training range -> [-1, 1], so the Fourier basis period is ~35 h for the
liver data) against the aligned scaling (24 h = 2 input units, so the basis
harmonics are 24, 12, 8 and 6 h). Writes a CSV and a Markdown table.

Usage: python scripts/compare_ablation.py --main results/v2/main --ablation results/v2/ablation_period --baselines results/v2/baselines.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_v2 import load_runs, MODELS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True); ap.add_argument("--ablation", required=True); ap.add_argument("--baselines")
    args = ap.parse_args()
    a = load_runs(args.main); b = load_runs(args.ablation)
    a = a[a.task.isin(["circadian_liver", "circadian_blood"])].copy(); a["encoding"] = "default"
    b = b.copy(); b["encoding"] = "24 h-aligned"; b["task"] = b.task.str.replace("_aligned", "", regex=False)
    df = pd.concat([a, b], ignore_index=True)
    rows = []
    for (task, model), g in df.groupby(["task", "model"]):
        rec = {"task": task, "model": model}
        for enc in ["default", "24 h-aligned"]:
            gg = g[g.encoding == enc]
            rec[f"inside {enc}"] = gg.inside_error.mean()
            if "outside_error" in gg and gg.outside_error.notna().any():
                rec[f"outside {enc}"] = gg.outside_error.mean()
        rows.append(rec)
    res = pd.DataFrame(rows)
    res["model"] = pd.Categorical(res.model, MODELS); res = res.sort_values(["task", "model"])
    out = Path(args.ablation)
    res.to_csv(out / "period_alignment.csv", index=False)
    base = pd.read_csv(args.baselines) if args.baselines else None
    lines = ["| Task | Model | Inside, default | Inside, 24 h-aligned | Outside (forecast), default | Outside, 24 h-aligned |", "|---|---|---:|---:|---:|---:|"]
    for _, r in res.iterrows():
        lines.append(f"| {r.task} | {r.model} | {r['inside default']:.3f} | {r['inside 24 h-aligned']:.3f} | "
                     + (f"{r['outside default']:.3f} | {r['outside 24 h-aligned']:.3f} |" if "outside default" in r and pd.notna(r.get("outside default")) else "— | — |"))
    if base is not None:
        for t in ["circadian_liver", "circadian_blood"]:
            bb = base[base.task == t]
            for split in ["inside", "outside"]:
                s = bb[bb.split == split]
                if len(s):
                    lines.append(f"| {t} | *reference: mean / ridge / cosinor ({split})* | {s['mean'].mean():.3f} | {s['ridge'].mean():.3f} | {s['cosinor'].mean():.3f} | |")
    (out / "period_alignment.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
