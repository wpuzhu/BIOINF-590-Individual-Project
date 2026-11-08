"""Model-agnostic structure audit: does each task have the structure we assumed?

The audit is run before any KAN/MLP comparison, with methods unrelated to both
model families, so the "type of data" claim is tested rather than assumed.
Synthetic targets are audited with the same code and act as positive controls.

Fingerprint (all R^2 values are 5-fold cross-validated, on pooled
train+val+inside data of one split; descriptive, never used for model selection):
  r2_linear      ridge regression on raw inputs
  r2_additive    additive spline model (sum of univariate cubic splines)
  r2_flexible    best of gradient boosting and RBF kernel ridge
  nonlinearity   r2_additive - r2_linear     (curvature that a GAM captures)
  interaction    r2_flexible - r2_additive   (signal that needs input mixing)
  shape          per continuous input: number of prominent interior extrema of
                 the 1-D smoothed marginal (0 monotone, 1 localized bump,
                 >=2 oscillatory) and the sigmoid-over-linear gain for monotone
                 marginals (threshold / saturation)
  periodicity    for time inputs: cosinor F-test of a 24 h rhythm on top of a
                 linear trend, per output, Benjamini-Hochberg FDR
For classification the same comparisons use McFadden pseudo-R^2 of
cross-validated predicted probabilities.
"""
import json
import warnings

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import GridSearchCV, KFold, GroupKFold, cross_val_predict, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

warnings.filterwarnings("ignore")


def _pool(data):
    keys = [k for k in ["train", "val", "inside"] if k in data]
    x = np.concatenate([data[k][0] for k in keys]).astype(float)
    y = np.concatenate([data[k][1] for k in keys])
    return x, y


def _r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def _models_reg(continuous):
    lin = RidgeCV(alphas=np.logspace(-4, 3, 15))
    add = make_pipeline(SplineTransformer(n_knots=8, degree=3), RidgeCV(alphas=np.logspace(-4, 3, 15))) if continuous else lin
    hgb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0)
    krr = GridSearchCV(KernelRidge(kernel="rbf"), {"alpha": [1e-3, 1e-2, 1e-1, 1], "gamma": [0.1, 0.3, 1, 3, 10]}, cv=3)
    return lin, add, [hgb, krr]


def _cv(groups, n):
    if groups is not None:
        return GroupKFold(n_splits=min(5, len(np.unique(groups))))
    return KFold(5, shuffle=True, random_state=0)


def regression_fingerprint(x, y, groups=None, continuous=True, max_n=3000):
    if len(x) > max_n:
        i = np.random.default_rng(0).choice(len(x), max_n, replace=False)
        x, y = x[i], y[i]; groups = None if groups is None else groups[i]
    xs = StandardScaler().fit_transform(x)
    cv = _cv(groups, len(x))
    out = {"r2_linear": [], "r2_additive": [], "r2_flexible": []}
    for j in range(y.shape[1]):
        yy = y[:, j]
        lin, add, flex = _models_reg(continuous)
        kw = {"groups": groups} if groups is not None else {}
        p_lin = cross_val_predict(lin, xs, yy, cv=cv, **kw)
        p_add = cross_val_predict(add, xs, yy, cv=cv, **kw)
        p_flex = max((_r2(yy, cross_val_predict(m, xs, yy, cv=cv, **kw)) for m in flex))
        out["r2_linear"].append(_r2(yy, p_lin)); out["r2_additive"].append(_r2(yy, p_add)); out["r2_flexible"].append(p_flex)
    res = {k: float(np.mean(v)) for k, v in out.items()}
    res["nonlinearity"] = res["r2_additive"] - res["r2_linear"]
    res["interaction"] = res["r2_flexible"] - res["r2_additive"]
    return res


def classification_fingerprint(x, y, continuous=True):
    xs = StandardScaler().fit_transform(x)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    k = int(y.max() + 1)
    prior = np.bincount(y, minlength=k) / len(y)
    null = -np.log(prior[y]).mean()

    def pseudo(model):
        p = cross_val_predict(model, xs, y, cv=cv, method="predict_proba")
        return 1 - (-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean()) / null

    lin = LogisticRegression(C=1.0, max_iter=3000)
    add = make_pipeline(SplineTransformer(n_knots=6, degree=3), LogisticRegression(C=0.3, max_iter=3000)) if continuous else lin
    flex = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                          l2_regularization=1.0, early_stopping=True, random_state=0)
    res = {"r2_linear": float(pseudo(lin)), "r2_additive": float(pseudo(add)), "r2_flexible": float(pseudo(flex))}
    res["nonlinearity"] = res["r2_additive"] - res["r2_linear"]
    res["interaction"] = res["r2_flexible"] - res["r2_additive"]
    return res


def _sigmoid(x, a, b, c, d):
    return a + b / (1 + np.exp(-c * (x - d)))


def _shape_1d(xj, yy, min_r2=0.10):
    """Smoothed 1-D fit of yy on xj; classify by prominent interior extrema."""
    model = make_pipeline(SplineTransformer(n_knots=6, degree=3), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    model.fit(xj[:, None], yy)
    r2_1d = float(_r2(yy, model.predict(xj[:, None])))
    if not np.isfinite(r2_1d) or r2_1d < min_r2:
        return {"kind": "flat", "extrema": None, "r2_marginal": r2_1d, "sigmoid_gain": None}
    grid = np.linspace(np.quantile(xj, 0.02), np.quantile(xj, 0.98), 200)
    f = model.predict(grid[:, None]); span = f.max() - f.min()
    # Zig-zag with 15 % hysteresis: a turn counts only after the curve has
    # moved back by at least 15 % of its fitted range.
    delta, ext, direction = 0.15 * span, 0, 0
    hi = lo = ref = f[0]
    for v in f[1:]:
        if direction == 0:
            hi, lo = max(hi, v), min(lo, v)
            if v - lo > delta: direction, ref = 1, v
            elif hi - v > delta: direction, ref = -1, v
        elif direction == 1:
            if v > ref: ref = v
            elif v < ref - delta: ext, direction, ref = ext + 1, -1, v
        else:
            if v < ref: ref = v
            elif v > ref + delta: ext, direction, ref = ext + 1, 1, v
    kind = "monotone" if ext == 0 else ("single-extremum" if ext == 1 else "oscillatory")
    gain = None
    if ext == 0 and span > 0:
        try:
            p0 = [f.min(), span * np.sign(f[-1] - f[0]), 4 / (grid[-1] - grid[0]), np.median(xj)]
            par, _ = curve_fit(_sigmoid, xj, yy, p0=p0, maxfev=20000)
            lin = np.polyfit(xj, yy, 1)
            gain = float(_r2(yy, _sigmoid(xj, *par)) - _r2(yy, np.polyval(lin, xj)))
        except Exception:
            gain = None
    return {"kind": kind, "extrema": int(ext), "r2_marginal": r2_1d, "sigmoid_gain": gain}


def marginal_shapes(x, y, max_n=3000, max_outputs=10):
    """Per continuous input, classify the 1-D marginal of every output (up to
    10) and report the most common non-flat shape. Outputs whose marginal
    explains <10 % of variance are 'flat' and carry no shape claim."""
    if len(x) > max_n:
        i = np.random.default_rng(0).choice(len(x), max_n, replace=False); x, y = x[i], y[i]
    y = y.reshape(len(y), -1)[:, :max_outputs]
    shapes = []
    for j in range(x.shape[1]):
        xj = x[:, j]
        if len(np.unique(xj)) < 6:
            shapes.append({"input": j, "kind": "discrete"}); continue
        per = [_shape_1d(xj, y[:, k]) for k in range(y.shape[1])]
        kinds = [p["kind"] for p in per if p["kind"] != "flat"]
        kind = max(set(kinds), key=kinds.count) if kinds else "flat"
        gains = [p["sigmoid_gain"] for p in per if p["sigmoid_gain"] is not None]
        shapes.append({"input": j, "kind": kind, "counts": {k: kinds.count(k) for k in set(kinds)},
                       "r2_marginal_max": float(max(p["r2_marginal"] for p in per)),
                       "sigmoid_gain": float(max(gains)) if gains else None})
    return shapes


def cosinor(t_hours, y, period=24.0, groups=None):
    """F-test: trend + cos/sin(2*pi*t/period) versus trend only, per output.
    Subject means are removed first when groups are given."""
    pvals, amps = [], []
    for j in range(y.shape[1]):
        yy = y[:, j].astype(float).copy()
        if groups is not None:
            for g in np.unique(groups):
                yy[groups == g] -= yy[groups == g].mean()
        w = 2 * np.pi * t_hours / period
        X0 = np.c_[np.ones_like(t_hours), t_hours]
        X1 = np.c_[X0, np.cos(w), np.sin(w)]
        r0 = yy - X0 @ np.linalg.lstsq(X0, yy, rcond=None)[0]
        b1 = np.linalg.lstsq(X1, yy, rcond=None)[0]; r1 = yy - X1 @ b1
        df1, df2 = 2, len(yy) - X1.shape[1]
        F = ((r0 @ r0 - r1 @ r1) / df1) / (r1 @ r1 / df2)
        pvals.append(float(stats.f.sf(F, df1, df2)))
        amps.append(float(np.hypot(b1[2], b1[3]) / (yy.std() + 1e-12)))
    p = np.array(pvals); order = np.argsort(p); m = len(p)
    q = np.empty(m); q[order] = np.minimum.accumulate((p[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    return {"period_h": period, "p": pvals, "q": q.tolist(), "rel_amplitude": amps,
            "fraction_rhythmic_q05": float((q < 0.05).mean())}


def audit(name, data, raw_time=None, groups=None):
    x, y = _pool(data)
    continuous = not set(np.unique(x)).issubset({-1.0, 1.0})
    if data["task"] == "classification":
        fp = classification_fingerprint(x, y, continuous)
        yy = np.eye(int(y.max() + 1))[y]
        if yy.shape[1] == 2:
            yy = yy[:, 1:]
    else:
        fp = regression_fingerprint(x, y, groups, continuous)
        yy = y
    fp["shapes"] = marginal_shapes(x, yy) if continuous else []
    if raw_time is not None:
        fp["periodicity"] = cosinor(raw_time, y, 24.0, groups)
    fp["n"] = int(len(x)); fp["inputs"] = int(x.shape[1]); fp["outputs"] = int(data["outputs"])
    return fp


def verdict(name, fp):
    """Turn a fingerprint into plain labels used in the report."""
    labels = []
    if fp["r2_flexible"] < 0.05:
        return ["little learnable signal"]
    if fp["nonlinearity"] < 0.02 and fp["interaction"] < 0.02:
        labels.append("near-linear")
    elif fp["nonlinearity"] >= 0.02:
        labels.append("nonlinear marginals")
    if fp["interaction"] >= 0.05:
        labels.append("interaction")
    elif fp["interaction"] >= 0.02 and "near-linear" not in labels:
        labels.append("weak interaction")
    kinds = [s["kind"] for s in fp.get("shapes", []) if s.get("kind") not in (None, "discrete")]
    if "oscillatory" in kinds:
        labels.append("oscillatory marginal")
    if "single-extremum" in kinds:
        labels.append("single-extremum (bump/U) marginal")
    gains = [s.get("sigmoid_gain") for s in fp.get("shapes", []) if s.get("sigmoid_gain") is not None]
    if gains and max(gains) > 0.02:
        labels.append("threshold/saturation")
    per = fp.get("periodicity")
    if per is not None:
        labels.append(f"24 h rhythm in {per['fraction_rhythmic_q05']:.0%} of outputs (q<0.05)")
    return labels


def to_json(obj):
    return json.loads(json.dumps(obj, default=float))
