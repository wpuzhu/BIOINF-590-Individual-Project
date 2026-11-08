"""Run the structure audit on every v2 task and write JSON + a Markdown table.

Usage: python scripts/run_audit.py --out results/v2/audit.json --doc docs/structure_audit.md
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kan_mlp import tasks  # noqa: E402
from kan_mlp.audit import audit, verdict, to_json  # noqa: E402


def time_and_groups(name, d):
    keys = [k for k in ["train", "val", "inside"] if k in d]
    if name in ("circadian_blood", "circadian_liver"):
        pre = d["preprocessing"]
        t = np.concatenate([(d[k][0][:, 0] + 1) * pre["time_half_range"] + pre["time_min"] for k in keys])
        g = np.concatenate([d["meta"]["row_subject"][k] for k in keys]) if name == "circadian_blood" else None
        return t, g
    return None, None


def modal_shapes(per_unit):
    """Most common shape per input across replicate units (ties -> monotone first)."""
    from collections import Counter
    out = []
    for j in range(len(per_unit[0])):
        kinds = Counter(u[j]["kind"] for u in per_unit)
        gains = [u[j].get("sigmoid_gain") for u in per_unit if u[j].get("sigmoid_gain") is not None]
        out.append({"input": j, "kind": kinds.most_common(1)[0][0], "counts": dict(kinds),
                    "sigmoid_gain": float(np.median(gains)) if gains else None})
    return out


def run(name, full_range=False):
    units = list(range(1, 13)) if name == "drug_combo" else [1]
    fps = []
    for u in units:
        d = tasks.load(name, u)
        if full_range and "outside" in d:  # audit the whole input range, incl. the extrapolation region
            d = dict(d); d["inside"] = tuple(np.concatenate([a, b]) for a, b in zip(d["inside"], d["outside"]))
        t, g = time_and_groups(name, d)
        fps.append(audit(name, d, t, g))
    if len(fps) == 1:
        return fps[0]
    keys = ["r2_linear", "r2_additive", "r2_flexible", "nonlinearity", "interaction"]
    out = {k: float(np.mean([f[k] for f in fps])) for k in keys}
    out.update({k + "_sd": float(np.std([f[k] for f in fps])) for k in keys})
    out["per_unit_shapes"] = [f["shapes"] for f in fps]
    out["shapes"] = modal_shapes(out["per_unit_shapes"])
    out["n"] = fps[0]["n"]; out["inputs"] = fps[0]["inputs"]; out["outputs"] = fps[0]["outputs"]; out["units"] = len(fps)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--doc", required=True)
    ap.add_argument("--doc-only", action="store_true", help="rewrite the Markdown from an existing JSON")
    args = ap.parse_args()
    if args.doc_only:
        results = json.loads(Path(args.out).read_text())
        for name, fp in results.items():
            if "per_unit_shapes" in fp:
                fp["shapes"] = modal_shapes(fp["per_unit_shapes"])
            fp["verdict"] = verdict(name, fp)
        Path(args.out).write_text(json.dumps(results, indent=2))
        write_doc(results, Path(args.doc)); return
    results = {}
    for name in tasks.HYPOTHESES:
        fp = to_json(run(name)); fp["verdict"] = verdict(name, fp); fp["hypothesis"] = tasks.HYPOTHESES[name]
        results[name] = fp
        print(name, fp["verdict"], flush=True)
    for name in ["gb1_epistasis", "drug_combo"]:
        fp = to_json(run(name, full_range=True)); fp["verdict"] = verdict(name, fp)
        fp["hypothesis"] = tasks.HYPOTHESES[name] + " [full range incl. outside region]"
        results[name + "_fullrange"] = fp
        print(name + "_fullrange", fp["verdict"], flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    write_doc(results, Path(args.doc))


def write_doc(results, path):
    rows = []
    for name, fp in results.items():
        sh = [s for s in fp.get("shapes", []) if s.get("kind") != "discrete"]
        if "per_unit_shapes" in fp:  # drug_combo: summarise both dose axes over all 12 blocks
            from collections import Counter
            shapes = "; ".join(f"x{j}: " + ", ".join(f"{k} {v}/{len(fp['per_unit_shapes'])}" for k, v in
                               Counter(u[j]["kind"] for u in fp["per_unit_shapes"]).most_common()) for j in range(fp["inputs"]))
        elif len(sh) > 4:
            from collections import Counter
            shapes = ", ".join(f"{k} x{v}" for k, v in Counter(s["kind"] for s in sh).most_common())
        else:
            shapes = ", ".join(f"x{s['input']}: {s['kind']}" for s in sh) or "one-hot (n/a)"
        rows.append(f"| {name} | {fp['hypothesis']} | {fp['n']} | {fp['r2_linear']:.2f} | {fp['r2_additive']:.2f} | "
                    f"{fp['r2_flexible']:.2f} | {fp['nonlinearity']:+.2f} | {fp['interaction']:+.2f} | {shapes} | {'; '.join(fp['verdict'])} |")
    table = "\n".join(rows)
    path.write_text(STRUCTURE_DOC.format(table=table))


STRUCTURE_DOC = open(Path(__file__).with_name("structure_audit_template.md")).read() if Path(__file__).with_name("structure_audit_template.md").exists() else "{table}"

if __name__ == "__main__":
    main()
