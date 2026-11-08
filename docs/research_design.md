# v2 research design: when does a KAN beat a parameter-matched MLP on biological data?

Branch `research-v2`. This document is the protocol; results are in [results_v2.md](results_v2.md), and the data-structure checks are in [structure_audit.md](structure_audit.md).

## Research questions

| # | Question | How it is tested |
|---|---|---|
| RQ1 | At an equal number of trainable parameters, do KANs (cubic-spline, Fourier or Gaussian-RBF edges) predict better than MLPs (tanh, ReLU, SiLU)? | 14 tasks × 6 models × 2 depths × 2 budgets × 5–12 replicates, paired by replicate |
| RQ2 | Does the advantage depend on the *structure* of the input–output map (periodic, localized, threshold, interaction)? | A model-free structure audit run before training, plus synthetic controls whose structure is known exactly; the KAN−MLP gap is related to the audited structure |
| RQ3 | Do learned KAN edge functions extrapolate better (forecast a rhythm, predict a higher dose, predict a more damaging mutation)? | Separate "outside" test splits that lie beyond the training input range |
| RQ4 | What does any gain cost, and does it depend on sample size? | Training time and multiply–accumulates (MACs) per sample at matched parameters; a sample-size sweep from 100 to 3,000 training points |

## Why these biological tasks

The v1 study used two classification datasets, yeast localization and breast cancer. Their structure was never checked, so a KAN win or loss could not be tied to the mechanism that motivates KANs. In v2 each biological task was chosen because biology predicts a specific shape for its input–output map. Each predicted shape matches one of the synthetic controls. The audit then checks whether the data actually show that shape.

| Task | Biological question | Inputs → output | Expected structure and why | Synthetic analogue | Split (what generalization means) |
|---|---|---|---|---|---|
| `circadian_liver` | Can a model learn and *forecast* the daily rhythm of liver clock genes? | Circadian time → log2 expression of 10 transcripts (Hughes 2009, hourly, 48 h) | **Periodic.** Clock-controlled transcription repeats every ~24 h; some genes carry 12 h harmonics | `waves` | Random 24/6/6 time points within CT18–53. *Outside* = CT54–65, i.e. forecasting the next 12 h |
| `circadian_blood` | Does the rhythm learned from some people transfer to new people? | Hours awake, protocol group → 10 blood transcripts (Möller-Levet 2013, 24 subjects) | **Periodic** within subjects, plus a homeostatic sleep-pressure trend and large between-person offsets | `waves` + `additive` | Grouped by subject 14/5/5, so the test set contains only unseen people |
| `drug_combo` | Can a model learn a two-drug dose–response surface and extrapolate to the top dose? | log10 dose A, log10 dose B → % inhibition (NCATS pyronaridine × artesunate, 12 replicate 10×10 blocks) | **Sigmoidal thresholds** (Hill curves) along each dose and **interaction** (synergy or antagonism) | `kink` + `interaction` | Within each block, inside wells 49/16/16. *Outside* = the highest-dose row and column (19 wells) |
| `gb1_epistasis` | Is the fitness of a double mutant predictable from its two single mutants (global epistasis)? | Single-mutant fitness A, B → double-mutant fitness (Olson 2014, GB1) | **Saturating / threshold interaction.** Fitness is a nonlinear readout of an additive stability scale, so effects compound near the unfolding threshold | `kink` / `radial` | Random 2,000/500/2,000 among pairs whose two singles are both above the 10th percentile. *Outside* = pairs that contain a strongly deleterious single |
| `mpsa_splicing` | Can 5′ splice-site strength be predicted from sequence? | One-hot 9-nt sequence (36 binary inputs) → log PSI (Wong 2018) | **High-order interactions** among positions, on binary inputs where per-edge curves have little room to help | `interaction` | The authors' own train and test sets |
| `diabetes` | Clinical progression from routine measurements | 10 clinical variables → 1-year progression | **Mostly additive and near-linear**: a negative control for KAN-specific benefit | `linear` / `additive` | Random 60/20/20 |
| `breast_cancer` | Tumour malignancy from nuclear morphology | 30 features → benign or malignant | Nearly linearly separable: a negative control | `linear` | Stratified 60/20/20 |
| `yeast` | Protein subcellular localization (v1 task, kept for continuity) | 8 sequence-derived scores → 10 classes | Weak, noisy structure | — | Stratified 60/20/20 |

**Is each task really the expected type of data?** That question is answered empirically in [structure_audit.md](structure_audit.md). The audit uses only methods unrelated to KANs and MLPs:
- cross-validated linear, additive-spline and flexible models;
- a 1-D shape classifier;
- a cosinor rhythm test.

The six synthetic targets go through the same audit as positive controls. Where the audit disagrees with the hypothesis, the task is reported under the structure the audit found. A KAN result on that task is never credited to the hypothesized mechanism. GB1 is the main example: inside the training range, double-mutant fitness is almost additive in the single-mutant fitnesses.

## Capacity matching ("same parameters")

Trainable parameters per layer from `a` inputs to `b` outputs, with `R = 8` basis terms:

| Family | Layer parameters | What they are |
|---|---|---|
| MLP | `a·b + b` | weight, bias |
| KAN | `(R + 2)·a·b + b = 10·a·b + b` | SiLU base weight, basis scale, 8 basis coefficients per edge, bias |

`kan_mlp.arch.architecture` searches the hidden widths for the count closest to the budget. Hidden layers share one width, and the last hidden layer may vary between half and double it. No frozen or dummy parameters are used. A configuration runs only if the count is within **±2 %** of the budget. v1 allowed 5 %. The largest deviation that actually occurred is reported with the results.

A KAN hidden unit costs about ten times as many parameters as an MLP unit, so a matched MLP is roughly ten times wider. For tasks with many inputs or outputs, a 1,024-parameter KAN cannot be matched within 2 %. Budgets are therefore set per task:

| Tasks | Small budget | Large budget |
|---|---:|---:|
| 1–2 inputs and 1 output: synthetic, `circadian_liver`, `drug_combo`, `gb1_epistasis` | 1,024 | 4,096 |
| Higher-dimensional or multi-output: `circadian_blood`, `mpsa_splicing`, `diabetes`, `breast_cancer`, `yeast` | 4,096 | 16,384 |

Equal parameter counts are **not** equal compute. A KAN edge evaluates 8 basis functions plus a SiLU path. Every run therefore also records MACs per sample, measured training time and prediction time.

## Fair training protocol (identical for every model)

- **Optimizer:** full-batch Adam with weight decay 1e-4 and gradient-norm clipping at 10.
- **Learning rate:** ReduceLROnPlateau (×0.5 after 3 non-improving checks) on validation loss.
- **Tuning:** the same three learning rates {0.001, 0.003, 0.01} for every model. The candidate with the lowest *validation* loss is kept.
- **Checks and stopping:** validation is checked every 10 epochs. Training stops after 12 checks without 0.1 % relative improvement, with a 3,000-epoch cap (v1: 1,500). The best validation checkpoint is restored.
- **No test leakage:** test splits are evaluated only after selection, and never influence stopping, learning rate or checkpoint choice.
- **Preprocessing:** scaling statistics come from training data only. Regression targets are standardized, so RMSE is in target-SD units. Time and dose inputs map the training range onto the KAN grid [−1, 1], so extrapolation splits fall outside the grid by design.
- **Shared data:** for a given replicate, every model sees the identical split.

## Statistical analysis

- **Error measure:** regression uses inside-test RMSE in target-SD units; classification uses 1 − macro-F1. Extrapolation splits are reported separately.
- **Primary comparison:** in each block (task, replicate, depth, budget), the KAN basis and the MLP activation are each chosen by **validation loss**. Their test errors are then compared. This is the comparison a practitioner could actually make, because it never picks the winner using test data.
- **Per task, depth and budget:**
  - mean paired difference with a t-based 95 % CI;
  - number of KAN wins;
  - two-sided Wilcoxon signed-rank test with Holm correction across all comparisons.

  With 5 replicates the smallest attainable Wilcoxon p is 0.0625, so the 5-replicate tasks cannot reach significance on their own. The CIs and the across-task pattern carry the inference.
- **All six models:** a Friedman test over (task, depth, budget) blocks, with seed-averaged errors, mean ranks and the Nemenyi critical difference.
- **RQ2:** Spearman correlation between the audited interaction and nonlinearity scores and the mean log(KAN/MLP error), across tasks. This is exploratory, with n = 14 tasks.
- **RQ4:** a sample-size sweep with 100, 300, 1,000 and 3,000 training points. It uses 2 hidden layers, 1,024 parameters and 3 seeds, on `waves`, `radial`, `kink`, `interaction` and `gb1_epistasis`.

## Implementation note: NumPy engine

The v2 runs used `kan_mlp/engine_np.py`, because the build environment had no PyTorch. The engine reimplements the repository's `models.py` and `study.py` with:
- the same tensors and initialization distributions;
- the same basis formulas;
- the same Adam, clipping, scheduler and stopping semantics.

Gradients are derived by hand. `tests/test_engine_np.py` checks them against finite differences, and, when PyTorch is installed (the CI workflow installs it), it also checks forward values and gradients against `models.py` to 1e-8. Random streams differ from PyTorch, so v1 and v2 numbers are not bit-identical; they are runs of the same models under the same protocol.

## Threats to validity

- **Fixed-grid KAN.** No grid extension, pruning or symbolic regression, so this is not the full pykan workflow. Hidden activations can leave [−1, 1], where only the SiLU path acts.
- **Small biological tasks.** `circadian_liver` has 24 training points and `drug_combo` 49 per block. These tasks probe the low-data regime. The replicate-level spread is reported, not hidden.
- **Ten-gene subsets.** The circadian datasets are the 10-transcript subsets distributed with MetaCycle, not genome-wide data.
- **Hyperparameter search.** The search covers the learning rate only. Other choices, such as grid size or weight decay, could favour either family.
- **Timing.** Timings come from a shared 2-core cloud machine and are only indicative.
