"""v2 experiment runner (NumPy engine, parallel, resumable).

One *run* = (task, replicate seed, depth, budget, model). Each run tries every
learning rate in the config with an identical training protocol, keeps the
candidate with the lowest validation loss, and only then evaluates on test
splits. Results are written as one JSON file per run, so interrupted studies
resume where they stopped.

Usage: python -m kan_mlp.run_v2 --config configs/v2_main.json --out results/v2/main --workers 2
"""
import argparse
import json
import os
import platform
import subprocess
import time
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss

from .arch import architecture, macs_per_sample
from . import engine_np as E
from . import tasks
from .data import ROOT


@lru_cache(maxsize=64)
def _data(name, seed, n_train):
    kw = {} if n_train is None else {"n_train": n_train}
    return tasks.load(name, seed, **kw)


def evaluate(model, data):
    out = {}
    for split in ["inside", "outside"]:
        if split not in data:
            continue
        x, y, truth = data[split]
        model(x[:8])
        times = []
        for _ in range(5):
            t = time.perf_counter(); pred = model(x); times.append(time.perf_counter() - t)
        m = {"predict_us_per_sample": float(np.median(times) * 1e6 / len(x)), "n": int(len(x))}
        if data["task"] == "regression":
            err = pred - y
            m["rmse"] = float(np.sqrt((err ** 2).mean()))
            m["clean_rmse"] = float(np.sqrt(((pred - truth) ** 2).mean()))
            ss = ((y - y.mean(0)) ** 2).sum(0)
            m["r2"] = float(np.mean(1 - (err ** 2).sum(0) / np.where(ss > 0, ss, np.nan)))
        else:
            z = pred - pred.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
            labels = p.argmax(1); k = data["outputs"]
            m.update(accuracy=float(accuracy_score(y, labels)), balanced_accuracy=float(balanced_accuracy_score(y, labels)),
                     macro_f1=float(f1_score(y, labels, labels=np.arange(k), average="macro", zero_division=0)),
                     log_loss=float(log_loss(y, p, labels=np.arange(k))))
        out[split] = m
    return out


def run_one(job):
    cfg, out = job["cfg"], Path(job["out"])
    path = out / "runs" / f"{job['id']}.json"
    if path.exists():
        return job["id"], "cached"
    data = _data(job["task"], job["seed"], job.get("n_train"))
    dims, p = architecture(job["model"], data["inputs"], data["outputs"], job["depth"], job["budget"], cfg["terms"])
    err = abs(p - job["budget"]) / job["budget"]
    if err > cfg["budget_tolerance"]:
        raise ValueError(f"{job['id']}: cannot match budget within tolerance ({p})")
    candidates, best_model, selected, cost = [], None, None, 0.0
    for lr in cfg["learning_rates"]:
        try:
            model, rec = E.fit(job["model"], dims, data, cfg, job["seed"] + 100000, lr)
        except FloatingPointError as e:  # diverged candidate: recorded, never selected
            candidates.append({"lr": lr, "error": str(e)}); continue
        cost += rec["fit_seconds"]; candidates.append(rec)
        if selected is None or rec["val_loss"] < selected["val_loss"]:
            best_model, selected = model, rec
    if selected is None:
        raise RuntimeError(f"{job['id']}: all learning rates diverged")
    # Test splits are consulted only after validation-only selection.
    result = {k: job[k] for k in ["task", "seed", "depth", "budget", "model"]}
    result.update(n_train=job.get("n_train"), dims=dims, parameters=p, budget_relative_error=err,
                  macs_per_sample=macs_per_sample(job["model"], dims, cfg["terms"]), search_seconds=cost,
                  selected={k: v for k, v in selected.items()}, candidates=candidates,
                  metrics=evaluate(best_model, data), task_type=data["task"])
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result)); tmp.replace(path)
    return job["id"], selected["stop_reason"]


def jobs_from_config(cfg, out):
    jobs = []
    for name, spec in cfg["tasks"].items():
        for n_train in spec.get("n_train", [None]):
            for seed in spec["seeds"]:
                for depth in spec.get("depths", cfg["depths"]):
                    for budget in spec["budgets"]:
                        for model in cfg["models"]:
                            tag = f"{name}_n{n_train}" if n_train else name
                            jobs.append({"id": f"{tag}_s{seed}_d{depth}_b{budget}_{model}", "task": name, "seed": seed,
                                         "depth": depth, "budget": budget, "model": model, "n_train": n_train,
                                         "cfg": cfg, "out": str(out)})
    return jobs


def _init_worker():
    os.environ["OMP_NUM_THREADS"] = "1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument("--limit", type=int, default=None, help="run only the first N jobs (smoke test)")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    out = Path(args.out); (out / "runs").mkdir(parents=True, exist_ok=True)
    saved = out / "config.json"
    if saved.exists() and json.loads(saved.read_text()) != cfg:
        raise ValueError("Output directory holds a different config; use a new directory")
    saved.write_text(json.dumps(cfg, indent=2))
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        commit = None
    (out / "environment.json").write_text(json.dumps({
        "engine": "numpy (kan_mlp.engine_np)", "python": platform.python_version(), "numpy": np.__version__,
        "platform": platform.platform(), "workers": args.workers, "blas_threads_per_worker": 1, "commit": commit}, indent=2))
    jobs = jobs_from_config(cfg, out)
    # Interleave expensive and cheap jobs so workers stay balanced.
    jobs.sort(key=lambda j: (j["seed"], j["task"], j["depth"], j["budget"], j["model"]))
    if args.limit:
        jobs = jobs[:args.limit]
    start = time.time(); done = 0
    with Pool(args.workers, initializer=_init_worker) as pool:
        for ident, status in pool.imap_unordered(run_one, jobs):
            done += 1
            if done % 25 == 0 or done == len(jobs):
                print(f"[{done}/{len(jobs)}] {time.time() - start:7.0f}s {ident} {status}", flush=True)


if __name__ == "__main__":
    main()
