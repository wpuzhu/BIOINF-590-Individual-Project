"""Robustness check: were epoch-capped runs disadvantaged?

Every main-study run that stopped at the 3,000-epoch cap is retrained at its
selected learning rate with a 15,000-epoch cap (same seed, same split, same
stopping rule). If the paired KAN-MLP conclusions changed, the cap would have
been biasing the comparison.

Usage: python scripts/convergence_check.py --runs results/v2/main --out results/v2/convergence_check.csv --workers 2
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kan_mlp import engine_np as E, tasks  # noqa: E402
from kan_mlp.run_v2 import evaluate  # noqa: E402


def redo(job):
    r, cfg = job
    d = tasks.load(r["task"], r["seed"])
    cfg = dict(cfg, max_epochs=15000)
    model, rec = E.fit(r["model"], r["dims"], d, cfg, r["seed"] + 100000, r["selected"]["lr"])
    m = evaluate(model, d)
    out = {k: r[k] for k in ["task", "seed", "depth", "budget", "model"]}
    out.update(lr=rec["lr"], epochs_3000cap=r["selected"]["epochs"], epochs_15000cap=rec["epochs"], stop_15000cap=rec["stop_reason"],
               val_3000cap=r["selected"]["val_loss"], val_15000cap=rec["val_loss"])
    for split in m:
        out[f"{split}_error_3000cap"] = r["metrics"][split]["clean_rmse"]
        out[f"{split}_error_15000cap"] = m[split]["clean_rmse"]
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    cfg = json.loads((Path(args.runs) / "config.json").read_text())
    jobs = []
    for p in sorted((Path(args.runs) / "runs").glob("*.json")):
        r = json.loads(p.read_text())
        if r["selected"]["stop_reason"] == "epoch_limit" and r["task_type"] == "regression":
            jobs.append((r, cfg))
    print(len(jobs), "epoch-capped runs", flush=True)
    with Pool(args.workers) as pool:
        rows = pool.map(redo, jobs)
    df = pd.DataFrame(rows); df.to_csv(args.out, index=False)
    print(df[["task", "model", "depth", "budget", "epochs_15000cap", "stop_15000cap", "inside_error_3000cap", "inside_error_15000cap"]].to_string())




def summarize(runs_dir, check_csv, out_md):
    """Recompute the validation-selected paired comparison with the extended-cap
    results substituted for the capped runs, for every affected task."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from analyze_v2 import load_runs, selected_pairs, task_headline
    df = load_runs(runs_dir); chk = pd.read_csv(check_csv)
    key = ["task", "seed", "depth", "budget", "model"]
    alt = df.merge(chk.drop(columns=["lr"]), on=key, how="left")
    hit = alt.val_15000cap.notna()
    alt.loc[hit, "val_loss"] = alt.loc[hit, "val_15000cap"]
    alt.loc[hit, "inside_error"] = alt.loc[hit, "inside_error_15000cap"]
    if "outside_error_15000cap" in alt:
        h2 = hit & alt.outside_error_15000cap.notna()
        alt.loc[h2, "outside_error"] = alt.loc[h2, "outside_error_15000cap"]
    tasks_ = sorted(chk.task.unique())
    lines = ["| Task | Capped runs | Inside ratio, 3,000-epoch cap | Inside ratio, capped runs retrained to 15,000 | KAN wins (before → after) |",
             "|---|---:|---|---|---|"]
    for t in tasks_:
        a = task_headline(selected_pairs(df[df.task == t])).iloc[0]
        b = task_headline(selected_pairs(alt[alt.task == t][df.columns])).iloc[0]
        lines.append(f"| {t} | {int((chk.task == t).sum())} | {np.exp(a.mean_log_ratio):.3f} ({np.exp(a.ci_low):.3f}–{np.exp(a.ci_high):.3f}) | "
                     f"{np.exp(b.mean_log_ratio):.3f} ({np.exp(b.ci_low):.3f}–{np.exp(b.ci_high):.3f}) | {int(a.kan_wins)}/{int(a.blocks)} → {int(b.kan_wins)}/{int(b.blocks)} |")
    Path(out_md).write_text("\n".join(lines) + "\n"); print("\n".join(lines))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--summarize":
        summarize(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        main()
