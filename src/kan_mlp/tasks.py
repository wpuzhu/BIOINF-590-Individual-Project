"""v2 task registry: synthetic controls plus biological tasks chosen for structure.

Every loader returns the same dictionary shape as data.synthetic():
  task, inputs, outputs, train/val/inside[/outside] = (x, y, truth), split_ids,
  preprocessing, plus `meta` describing the structural hypothesis and split.

Rules applied everywhere:
  * all scaling statistics come from the training split only;
  * regression targets are standardised with training statistics, so RMSE is
    in training-SD units and comparable across outputs and tasks;
  * time / dose inputs are min-max scaled so the training range maps to
    [-1, 1] (the KAN basis grid), and extrapolation splits fall outside it;
  * grouped data (subjects, drug-screen blocks) are split by group.
"""
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from .data import ROOT, TARGETS, synthetic, biological

PROCESSED = ROOT / "data" / "processed"

# Structural hypothesis per task: what we expect the input-output map to look like.
HYPOTHESES = {
    "linear": "control: linear",
    "additive": "control: additive, non-monotone in x1",
    "interaction": "control: pure product interaction",
    "waves": "control: periodic interaction",
    "radial": "control: localized bump",
    "kink": "control: threshold (piecewise linear)",
    "circadian_blood": "periodic in time (circadian, ~24 h) plus homeostatic trend",
    "circadian_liver": "periodic in time (circadian, 24 h, sharp harmonics for some genes)",
    "drug_combo": "sigmoidal dose thresholds with drug-drug interaction",
    "gb1_epistasis": "threshold / saturating global epistasis with interaction",
    "mpsa_splicing": "high-order sequence interactions (one-hot input)",
    "diabetes": "mostly additive clinical regression",
    "breast_cancer": "mostly linear-separable morphology classification",
    "yeast": "protein-feature classification, weak structure",
}
ABLATIONS = {
    "circadian_liver_aligned": "circadian_liver with time scaled so 24 h = 2 input units (Fourier basis period)",
    "circadian_blood_aligned": "circadian_blood with time scaled so 24 h = 2 input units (Fourier basis period)",
}


def _ids(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def _minmax(train_col):
    lo, hi = float(np.min(train_col)), float(np.max(train_col))
    return lo, (hi - lo) / 2 if hi > lo else 1.0


def _pack(task, xs, ys, meta, scaler=None, truths=None):
    """xs, ys: dict split -> arrays (already input-scaled). Standardises y."""
    data = {"task": task, "inputs": int(xs["train"].shape[1]), "meta": meta, "split_ids": {}}
    if task == "regression":
        mu, sd = ys["train"].mean(0), ys["train"].std(0) + 1e-8
        data["outputs"] = int(ys["train"].shape[1])
        data["preprocessing"] = {"y_mean": mu.tolist(), "y_sd": sd.tolist(), **(scaler or {})}
        for k in xs:
            y = ((ys[k] - mu) / sd).astype("float32")
            t = y if truths is None else ((truths[k] - mu) / sd).astype("float32")
            data[k] = (xs[k].astype("float32"), y, t)
            data["split_ids"][k] = _ids(xs[k])
    else:
        data["outputs"] = int(max(ys[k].max() for k in ys) + 1)
        data["preprocessing"] = scaler or {}
        for k in xs:
            data[k] = (xs[k].astype("float32"), ys[k].astype("int64"), ys[k].astype("int64"))
            data["split_ids"][k] = _ids(xs[k])
    return data


# ------------------------------------------------------------------ circadian
def circadian_blood(seed, aligned=False):
    """Moller-Levet et al. 2013: 24 subjects, 3-hourly blood sampling over a
    ~27 h constant-routine window. Inputs: hours awake (min-max scaled) and
    protocol group (+/-1). Outputs: 10 transcripts. Split by subject so the
    test set contains only unseen people (14 train / 5 val / 5 test)."""
    df = pd.read_csv(PROCESSED / "circadian_blood.csv.gz")
    genes = list(df.columns[4:])
    subjects = np.array(sorted(df.subject.unique()))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(subjects)
    groups = {"train": perm[:14], "val": perm[14:19], "inside": perm[19:]}
    lo, half = _minmax(df[df.subject.isin(groups["train"])].time_hoursawake)
    if aligned:  # 24 h = 2 input units: the Fourier basis (period 2) then has 24/12/8/6 h harmonics
        lo, half = lo + half - 12.0, 12.0
    xs, ys = {}, {}
    for k, subj in groups.items():
        d = df[df.subject.isin(subj)]
        t = (d.time_hoursawake.values - lo) / half - 1
        g = np.where(d.group.values == "SleepRestriction", 1.0, -1.0)
        xs[k] = np.c_[t, g]; ys[k] = d[genes].values.astype(float)
    rows = {k: df[df.subject.isin(subj)].subject.tolist() for k, subj in groups.items()}
    meta = {"split": "grouped by subject 14/5/5", "time_column": 0, "time_unit_per_input": half, "row_subject": rows,
            "period_hint_hours": 24.0, "genes": genes, "groups": {k: v.tolist() for k, v in groups.items()}}
    return _pack("regression", xs, ys, meta, {"time_min": lo, "time_half_range": half})


def circadian_liver(seed, aligned=False):
    """Hughes et al. 2009: mouse liver, hourly CT18-CT65 (48 points), 10
    transcripts. Inside: random held-out points within CT18-CT53. Outside:
    the final 12 hours (CT54-CT65), i.e. forecasting the rhythm forward."""
    df = pd.read_csv(PROCESSED / "circadian_liver.csv.gz")
    genes = list(df.columns[1:])
    early, late = df[df.ct <= 53], df[df.ct > 53]
    idx = np.random.default_rng(seed).permutation(len(early))
    parts = {"train": early.iloc[idx[:24]], "val": early.iloc[idx[24:30]], "inside": early.iloc[idx[30:]], "outside": late}
    lo, half = _minmax(parts["train"].ct)
    if aligned:  # 24 h = 2 input units (Fourier harmonics at 24, 12, 8, 6 h); range exceeds [-1, 1]
        lo, half = lo + half - 12.0, 12.0
    xs = {k: ((v.ct.values - lo) / half - 1)[:, None] for k, v in parts.items()}
    ys = {k: np.log2(v[genes].values.astype(float)) for k, v in parts.items()}
    meta = {"split": "random 24/6/6 within CT18-53; outside = CT54-65", "time_column": 0,
            "time_unit_per_input": half, "period_hint_hours": 24.0, "genes": genes}
    return _pack("regression", xs, ys, meta, {"time_min": lo, "time_half_range": half})


# -------------------------------------------------------------- drug combos
def drug_combo(block):
    """NCATS 10x10 pyronaridine x artesunate matrices (12 replicate blocks).
    The replicate unit is the block (seed = block id 1-12). Inputs: log10 dose
    of each drug (zero dose placed one 2-fold dilution below the lowest dose).
    Outside: every well where either drug is at its highest dose (19 wells),
    i.e. extrapolating the dose-response surface. Inside wells 49/16/16."""
    df = pd.read_csv(PROCESSED / "drug_combo.csv.gz")
    d = df[df.block == int(block)].copy()
    for c in ["conc_a_uM", "conc_b_uM"]:
        pos = d[c][d[c] > 0]
        d[c + "_log"] = np.log10(d[c].where(d[c] > 0, pos.min() / 2))
    top = (d.row_index == 10) | (d.col_index == 10)
    inner, outer = d[~top], d[top]
    tr, rest = train_test_split(np.arange(len(inner)), train_size=49, random_state=int(block))
    va, te = train_test_split(rest, test_size=16, random_state=int(block) + 100)
    parts = {"train": inner.iloc[tr], "val": inner.iloc[va], "inside": inner.iloc[te], "outside": outer}
    cols = ["conc_a_uM_log", "conc_b_uM_log"]
    stats = {c: _minmax(parts["train"][c]) for c in cols}
    xs = {k: np.c_[[(v[c].values - stats[c][0]) / stats[c][1] - 1 for c in cols]].T for k, v in parts.items()}
    ys = {k: v[["inhibition"]].values.astype(float) for k, v in parts.items()}
    meta = {"split": "within block: top-dose row/column held out as outside; inside 49/16/16",
            "block": int(block), "drugs": [d.drug_a.iloc[0], d.drug_b.iloc[0]]}
    return _pack("regression", xs, ys, meta, {c: list(stats[c]) for c in cols})


# --------------------------------------------------------------------- GB1
def gb1_epistasis(seed, n_train=2000):
    """Olson et al. 2014 GB1 double mutants. Inputs: the two single-mutant
    log-enrichments; output: the double-mutant log-enrichment. This is the
    classic global-epistasis view: if effects were additive, y_ab = y_a + y_b.
    Outside: pairs containing a strongly deleterious single (below the 10th
    percentile of single fitness), never seen in training."""
    df = pd.read_csv(PROCESSED / "gb1_epistasis.csv.gz")
    singles = np.r_[df.fitness_a.values, df.fitness_b.values]
    cut = float(np.quantile(singles, 0.10))
    low = np.minimum(df.fitness_a, df.fitness_b) < cut
    inner, outer = df[~low], df[low]
    rng = np.random.default_rng(seed)
    i = rng.permutation(len(inner)); o = rng.permutation(len(outer))
    parts = {"train": inner.iloc[i[:n_train]], "val": inner.iloc[i[n_train:n_train + 500]],
             "inside": inner.iloc[i[n_train + 500:n_train + 2500]], "outside": outer.iloc[o[:2000]]}
    mu = parts["train"][["fitness_a", "fitness_b"]].values.mean(0)
    sd = parts["train"][["fitness_a", "fitness_b"]].values.std(0)
    xs = {k: (v[["fitness_a", "fitness_b"]].values - mu) / sd for k, v in parts.items()}
    ys = {k: v[["fitness_ab"]].values.astype(float) for k, v in parts.items()}
    meta = {"split": f"random {n_train}/500/2000 among pairs with both singles >= q10; outside = pairs with a single < q10",
            "q10_single_fitness": cut}
    return _pack("regression", xs, ys, meta, {"x_mean": mu.tolist(), "x_sd": sd.tolist()})


# -------------------------------------------------------------------- MPSA
def mpsa_splicing(seed, n_train=3000):
    """Wong et al. 2018 5' splice-site library: 9-nt sequence -> log PSI.
    One-hot input (36 binary features). Train/val sampled from the authors'
    training+validation sets, test from their held-out test set."""
    df = pd.read_csv(PROCESSED / "mpsa_splicing.csv.gz")
    rng = np.random.default_rng(seed)
    pool = df[df.source_split != "test"]; test = df[df.source_split == "test"]
    i = rng.permutation(len(pool)); t = rng.permutation(len(test))
    parts = {"train": pool.iloc[i[:n_train]], "val": pool.iloc[i[n_train:n_train + 1000]], "inside": test.iloc[t[:3000]]}
    alphabet = "ACGU"
    def onehot(seqs):
        a = np.array([[alphabet.index(c) for c in s] for s in seqs])
        out = np.zeros((len(a), 9 * 4)); out[np.arange(len(a))[:, None], np.arange(9) * 4 + a] = 1
        return out * 2 - 1  # {-1, +1}: centred and inside the basis grid
    xs = {k: onehot(v.sequence) for k, v in parts.items()}
    ys = {k: v[["log_psi"]].values.astype(float) for k, v in parts.items()}
    meta = {"split": f"train/val {n_train}/1000 from source train+val; test 3000 from source test", "encoding": "one-hot {-1,+1}"}
    return _pack("regression", xs, ys, meta)


# ----------------------------------------------------------------- diabetes
def diabetes(seed):
    """Efron et al. 2004 diabetes progression (n=442, 10 clinical inputs). 60/20/20."""
    ds = load_diabetes(scaled=False)
    x, y = ds.data, ds.target[:, None]
    tr, hold = train_test_split(np.arange(len(y)), test_size=0.4, random_state=seed)
    va, te = train_test_split(hold, test_size=0.5, random_state=seed + 10000)
    mu, sd = x[tr].mean(0), x[tr].std(0)
    xs = {k: (x[i] - mu) / sd for k, i in [("train", tr), ("val", va), ("inside", te)]}
    ys = {k: y[i].astype(float) for k, i in [("train", tr), ("val", va), ("inside", te)]}
    return _pack("regression", xs, ys, {"split": "random 60/20/20"}, {"x_mean": mu.tolist(), "x_sd": sd.tolist()})


# -------------------------------------------------------------- dispatcher
def _synthetic(name, seed, n_train=1000, noise=0.05):
    d = synthetic(name, seed, n_train, noise)
    d["meta"] = {"split": "uniform [-1,1]^2; outside = [-1.5,1.5]^2 minus [-1,1]^2"}
    return d


def _classification(name, seed):
    d = biological(name, seed)
    d["meta"] = {"split": "stratified 60/20/20"}
    return d


LOADERS = {
    **{k: _synthetic for k in TARGETS},
    "circadian_blood": lambda n, s, **kw: circadian_blood(s),
    "circadian_liver": lambda n, s, **kw: circadian_liver(s),
    "circadian_liver_aligned": lambda n, s, **kw: circadian_liver(s, aligned=True),
    "circadian_blood_aligned": lambda n, s, **kw: circadian_blood(s, aligned=True),
    "drug_combo": lambda n, s, **kw: drug_combo(s),
    "gb1_epistasis": lambda n, s, n_train=2000, **kw: gb1_epistasis(s, n_train),
    "mpsa_splicing": lambda n, s, n_train=3000, **kw: mpsa_splicing(s, n_train),
    "diabetes": lambda n, s, **kw: diabetes(s),
    "breast_cancer": lambda n, s, **kw: _classification(n, s),
    "yeast": lambda n, s, **kw: _classification(n, s),
}


def load(name, seed, **kw):
    if name in TARGETS:
        return _synthetic(name, seed, **kw)
    return LOADERS[name](name, seed, **kw)
