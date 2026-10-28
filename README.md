# BIOINF 590 Individual Project — KAN versus MLP

**How do target structure, learned basis functions and depth change predictive accuracy and computational cost at comparable parameter budgets?**

This repository contains the original interactive browser lab, a PyTorch simulation pipeline, a focused literature review, biological dataset loaders, tests and measured experiment results. The Python study compares **MLP tanh/ReLU/SiLU** with **KAN SiLU + cubic spline/Fourier/Gaussian RBF**. It is not restricted to one hidden layer.

## Start here

- [Interactive simulation lab](interactive/index.html): open locally in a browser; actual JavaScript training, four teaching sections and inside/outside test comparisons.
- [Literature review and biological task mapping](docs/literature_review.md).
- [Model formulas, algorithms and study protocol](docs/algorithms.md).
- [Dataset cards and proposed periodic/localized extensions](docs/datasets.md).
- [Measured results and limitations](docs/results.md).

The HTML is preserved from the prior teaching lab. It supports a shallow educational comparison; **the Python pipeline is the research implementation**, with different capacity matching and training details. Browser and Python results are not interchangeable.

## Install and verify

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
```

Python >=3.10 is required. Experiments run on CPU with one Torch thread. `requirements-tested.txt` records the versions used here; `pyproject.toml` gives installation requirements. This is a tested environment record, not a cross-platform lockfile.

## Reproduce the main study

```bash
python -m kan_mlp.study --config configs/simulation.json --out results/simulation
python scripts/summarize.py results/simulation
python -m kan_mlp.study --config configs/biology.json --out results/biology
python scripts/summarize.py results/biology
```

Run the studies sequentially for timing comparisons. Existing completed model runs resume automatically only when the config matches. Use a new output directory for a changed config. Pilot configs have a 300-epoch cap; main configs use 1,500 epochs and three shared seeds. Full configs provide **1/2/3 hidden layers, two budgets, five seeds and a 5,000-epoch cap**:

```bash
python -m kan_mlp.study --config configs/simulation_full.json --out results/simulation_full
python -m kan_mlp.study --config configs/biology_full.json --out results/biology_full
```

The full configs are provided for extension; their results are not claimed in this repository unless the corresponding output directory exists. Larger runs can take substantially longer. Runtime is implementation- and machine-specific.

## Experiment design

| Axis | Protocol |
|---|---|
| Synthetic targets | Linear, smooth additive, product interaction, periodic interaction, localized bump, piecewise-linear ridge; formulas match the HTML |
| Models | Three fixed-activation MLPs and three SiLU-plus-basis KANs |
| Depth | Main runs: 1 and 2 hidden layers; full configs: 1, 2 and 3 |
| Capacity | Active parameter counts within 5% of a shared budget; actual counts and widths saved; no dummy parameters |
| Data | Simulation: 1,000 train + 400 validation + 1,000 inside test + 1,000 outside test |
| Test regions | Inside: [-1,1]^2; outside: [-1.5,1.5]^2 minus [-1,1]^2 |
| Optimization | Adam; equal candidate LRs; validation plateau stopping, LR reductions and best-checkpoint restoration |
| Selection | Validation-only LR/checkpoint selection; test sets never control stopping |
| Biology | UCI Yeast protein localization and Wisconsin Diagnostic Breast Cancer morphology; stratified internal held-out testing |
| Output | Prediction error, classification metrics, training/search time, epochs, stop reason, prediction throughput and learning curves |

**A plateau is not proof of full model capacity.** Results explicitly distinguish `validation_plateau` from `epoch_limit`. Equal parameter budgets do not equalize function classes, operations or identifiable degrees of freedom. The fixed-grid KAN is a controlled implementation, not the complete adaptive-grid/pruning/symbolic workflow of the original paper.

## Biological questions

- **Periodic:** predict cell-cycle or circadian gene expression from time and condition. Spellman et al. and GEO GSE54650 provide relevant measured systems; use time-block and replicate-aware splits.
- **Localized:** predict dose-window or transient responses from concentration and time; alternatively, use spatial expression with coordinate-block and donor-level validation.
- **Interactions:** epistasis or drug-pair response with genotype/pair-aware held-out groups.
- **Thresholds:** regulatory switches, comparing ReLU's sharp ridge with smooth bases.

These are proposed mechanism-aligned follow-ups. The completed real-data benchmarks test biological classification; they do not validate periodicity or localized-response hypotheses. Protein localization classes should not be confused with local response curves.

## Raw experiment records

The complete per-run histories, split records, pilot results and all figures are in [results/records.zip](results/records.zip). Summary tables and the main learning curves are also browsable directly. Extract the records before inspecting raw run JSON or re-running the summary scripts:

```bash
python -m zipfile -e results/records.zip .
```

## Files and reproducibility

```text
interactive/       Original offline teaching lab
src/kan_mlp/       Models, split generation, training and evaluation
configs/           Pilot, main and full JSON protocols
scripts/           Config generation and results aggregation/plots
results/           Per-model JSON, configs, splits, metrics and figures
                     (.pt checkpoints are generated locally and ignored by Git)
data/              Unmodified Yeast data, original metadata and checksums
docs/              Literature review, algorithms, dataset cards and results
tests/             Gradient, parameter-count, split and checkpoint checks
.github/workflows/ Automated test workflow
```

Each run saves all LR candidates' histories, the selected checkpoint metadata, split IDs/fingerprints, training-only scaler parameters, environment versions and actual parameter counts. Results can be recomputed from source. Committed results omit checkpoints; re-run to regenerate weights. Some initial environment records have a null Git commit because the repository had not yet been initialized when they started.

## Data attribution and reuse

Yeast: Nakai, K. (1991), UCI Machine Learning Repository, https://doi.org/10.24432/C5KG68, CC BY 4.0. Original files are unmodified. Wisconsin Diagnostic Breast Cancer is loaded through scikit-learn; the source and original work are credited in [dataset cards](docs/datasets.md). Dataset terms remain separate from project code. Review notes paraphrase and link papers; full publisher PDFs are not redistributed.

Project code has no license grant specified yet. Choose a license before third-party redistribution. This project is an educational research comparison and has no clinical deployment validation.
