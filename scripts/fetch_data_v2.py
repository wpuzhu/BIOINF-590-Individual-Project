"""Build the processed v2 biological datasets from pinned public sources.

Sources are fetched with `git` from GitHub at fixed commits, so the build is
reproducible without R. Each processed file is written to data/processed/ and
recorded (sha256, source, commit, citation) in data/processed/manifest_v2.json.

Usage:  python scripts/fetch_data_v2.py [--cache DIR]
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kan_mlp import rdata_min  # noqa: E402

OUT = ROOT / "data" / "processed"
SOURCES = {
    "MetaCycle": ("https://github.com/gangwug/MetaCycle", "fad6d443e8407d44983d1f9fe73976d5e90fb4da"),
    "synergyfinder": ("https://github.com/shuyuzheng/synergyfinder", "ea2d246e4e14c79767df90eec67b9a2b75162b92"),
    "mavenn": ("https://github.com/jbkinney/mavenn", "95b5ff4913b961ac172fbc23c22be986f9f671d7"),
}
FILES = {
    "MetaCycle": ["data/cycHumanBloodData.rda", "data/cycHumanBloodDesign.rda", "data/cycMouseLiverRNA.rda"],
    "synergyfinder": ["data-raw/NCATS_10023/responses.csv", "data-raw/NCATS_10023/metadata.csv"],
    "mavenn": ["paper/datasets/gb1_single_and_double_data.csv.gz", "mavenn/examples/datasets/mpsa_data.csv.gz"],
}


def fetch(cache):
    paths = {}
    for name, (url, commit) in SOURCES.items():
        repo = cache / name
        if not (repo / ".git").exists():
            subprocess.run(["git", "clone", "-q", "--filter=blob:none", "--no-checkout", url, str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "fetch", "-q", "--depth", "1", "origin", commit], check=False)
        subprocess.run(["git", "-C", str(repo), "checkout", "-q", commit, "--", *FILES[name]], check=True)
        paths[name] = repo
    return paths


def frame(path, key):
    return rdata_min.as_frame(rdata_min.read(path)[key])


def build_circadian_blood(src):
    data = frame(src / "data/cycHumanBloodData.rda", "cycHumanBloodData").set_index("ID_REF").T
    design = frame(src / "data/cycHumanBloodDesign.rda", "cycHumanBloodDesign").set_index("sample_library")
    df = design.join(data, how="inner").reset_index().rename(columns={"index": "sample"})
    return df


def build_circadian_liver(src):
    d = frame(src / "data/cycMouseLiverRNA.rda", "cycMouseLiverRNA").set_index("geneName").T
    d.insert(0, "ct", [float(c[2:]) for c in d.index])
    return d.reset_index(drop=True)


def build_drug_combo(src):
    r = pd.read_csv(src / "data-raw/NCATS_10023/responses.csv")
    m = pd.read_csv(src / "data-raw/NCATS_10023/metadata.csv").set_index("BlockId")
    rows = []
    for _, x in r.iterrows():
        meta = m.loc[x.BlockId]
        # Concentration lists are stored high->low; matrix index 1 is the lowest
        # dose (0). Verified: mean response increases monotonically with index.
        rc = [float(v) for v in meta.RowConcs.split(",")][::-1]
        cc = [float(v) for v in meta.ColConcs.split(",")][::-1]
        rows.append({"block": int(x.BlockId), "drug_a": meta.RowName, "drug_b": meta.ColName,
                     "conc_a_uM": rc[int(x.Row) - 1], "conc_b_uM": cc[int(x.Col) - 1],
                     "row_index": int(x.Row), "col_index": int(x.Col), "inhibition": float(x.Value)})
    return pd.DataFrame(rows)


def build_gb1(src, min_count=100):
    g = pd.read_csv(src / "paper/datasets/gb1_single_and_double_data.csv.gz")
    # Wild type = per-position consensus of all single mutants (before filtering);
    # equals the GB1 domain sequence QYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE.
    all_singles = g[g.dist == 1]
    length = len(all_singles.x.iloc[0])
    arr = np.array([list(s) for s in all_singles.x])
    wt = "".join(pd.Series(arr[:, k]).mode()[0] for k in range(length))
    g = g[g.input_ct >= min_count]
    singles = g[g.dist == 1]

    def muts(seq):
        return [f"{wt[k]}{k + 2}{seq[k]}" for k in range(length) if seq[k] != wt[k]]

    # The wild-type row is stored with dist == 1 and y == 0 (the reference); skip it.
    single_fit = {muts(s)[0]: y for s, y in zip(singles.x, singles.y) if s != wt}
    out = []
    for s, y, n in zip(g.x[g.dist == 2], g.y[g.dist == 2], g.input_ct[g.dist == 2]):
        a, b = muts(s)
        if a in single_fit and b in single_fit:
            out.append((a, b, single_fit[a], single_fit[b], y, n))
    df = pd.DataFrame(out, columns=["mut_a", "mut_b", "fitness_a", "fitness_b", "fitness_ab", "input_count"])
    # A fixed random subsample keeps the committed file small (<1 MB); 50,000
    # doubles are far more than any split uses.
    df = df.sample(n=min(50000, len(df)), random_state=0).reset_index(drop=True)
    return df, wt


def build_mpsa(src):
    d = pd.read_csv(src / "mavenn/examples/datasets/mpsa_data.csv.gz")
    return d[["x", "y", "set", "tot_ct"]].rename(columns={"x": "sequence", "y": "log_psi", "set": "source_split"})


CITATIONS = {
    "circadian_blood": ("Moller-Levet CS et al. (2013) PNAS 110:E1132-41; GEO GSE39445; 10-transcript subset distributed with MetaCycle (GPL>=2).", "MetaCycle"),
    "circadian_liver": ("Hughes ME et al. (2009) PLoS Genet 5:e1000442; 10-transcript subset distributed with MetaCycle (GPL>=2).", "MetaCycle"),
    "drug_combo": ("NCATS matrix screen, assay 10023 (pyronaridine x artesunate, 12 blocks), distributed with synergyfinder (MPL-2.0).", "synergyfinder"),
    "gb1_epistasis": ("Olson CA, Wu NC, Sun R (2014) Curr Biol 24:2643-51; processed data distributed with MAVE-NN (MIT).", "mavenn"),
    "mpsa_splicing": ("Wong MS et al. (2018) Mol Cell 71:1012-26; processed data distributed with MAVE-NN (MIT).", "mavenn"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(ROOT / ".cache_sources"))
    args = ap.parse_args()
    cache = Path(args.cache); cache.mkdir(parents=True, exist_ok=True)
    src = fetch(cache)
    OUT.mkdir(parents=True, exist_ok=True)
    gb1, wt = build_gb1(src["mavenn"])
    tables = {
        "circadian_blood": build_circadian_blood(src["MetaCycle"]),
        "circadian_liver": build_circadian_liver(src["MetaCycle"]),
        "drug_combo": build_drug_combo(src["synergyfinder"]),
        "gb1_epistasis": gb1,
        "mpsa_splicing": build_mpsa(src["mavenn"]),
    }
    manifest = {}
    for name, df in tables.items():
        path = OUT / f"{name}.csv.gz"
        df.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
        cite, repo = CITATIONS[name]
        manifest[name] = {"file": str(path.relative_to(ROOT)), "rows": int(len(df)), "columns": list(map(str, df.columns)),
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "source_repo": SOURCES[repo][0],
                          "source_commit": SOURCES[repo][1], "source_files": FILES[repo], "citation": cite}
        print(name, df.shape)
    manifest["gb1_epistasis"]["wild_type"] = wt
    manifest["gb1_epistasis"]["filter"] = ("input_count >= 100 for singles and doubles; doubles kept only if both singles pass; "
                                           "fixed random subsample of 50,000 doubles (random_state=0)")
    (OUT / "manifest_v2.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
