"""Reference predictors for every v2 regression task and replicate.

  mean      predict the training mean (0 in standardised units): RMSE ~ 1 means "no better than nothing"
  ridge     linear ridge regression on the same inputs
  cosinor   circadian tasks only: per transcript, linear trend + 24 h and 12 h harmonics
            fitted on training data (the parametric model a chronobiologist would use)

Usage: python scripts/baselines_v2.py --config configs/v2_main.json --config configs/v2_ablation_period.json --out results/v2/baselines.csv
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kan_mlp import tasks  # noqa: E402


def rmse(p, y):
    return float(np.sqrt(((p - y) ** 2).mean()))


def hours(d, x):
    pre = d["preprocessing"]
    return (x[:, 0] + 1) * pre["time_half_range"] + pre["time_min"]


def cosinor_design(t):
    w = 2 * np.pi * t / 24.0
    return np.c_[np.ones_like(t), t, np.cos(w), np.sin(w), np.cos(2 * w), np.sin(2 * w)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", action="append", required=True); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = []
    for cfg_path in args.config:
        cfg = json.loads(Path(cfg_path).read_text())
        for name, spec in cfg["tasks"].items():
            for seed in spec["seeds"]:
                d = tasks.load(name, seed)
                if d["task"] != "regression":
                    continue
                xtr, ytr, _ = d["train"]
                xfit = np.r_[xtr, d["val"][0]]; yfit = np.r_[ytr, d["val"][1]]
                ridge = RidgeCV(alphas=np.logspace(-4, 3, 15)).fit(xfit, yfit)
                for split in ["inside", "outside"]:
                    if split not in d:
                        continue
                    x, y, truth = d[split]
                    rec = {"task": name, "seed": seed, "split": split,
                           "mean": rmse(np.zeros_like(truth), truth), "ridge": rmse(ridge.predict(x).reshape(truth.shape), truth)}
                    if name.startswith("circadian"):
                        X = cosinor_design(hours(d, xfit))
                        if name.startswith("circadian_blood"):  # add the protocol-group indicator
                            X = np.c_[X, xfit[:, 1]]
                        beta = np.linalg.lstsq(X, yfit, rcond=None)[0]
                        Xt = cosinor_design(hours(d, x))
                        if name.startswith("circadian_blood"):
                            Xt = np.c_[Xt, x[:, 1]]
                        rec["cosinor"] = rmse(Xt @ beta, truth)
                    rows.append(rec)
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.groupby(["task", "split"]).mean(numeric_only=True).drop(columns="seed").round(3).to_string())


if __name__ == "__main__":
    main()
