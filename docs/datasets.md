# Dataset cards and provenance

## Synthetic function-recovery suite

The six formulas exactly match `interactive/index.html`: linear, additive, interaction, waves, radial and kink. `src/kan_mlp/data.py` generates them deterministically. Each seed defines 1,000 training, 400 validation, 1,000 inside-test and 1,000 outside-test points. Noise is Gaussian SD 0.05. Input/test fingerprints are saved. Ground truth is available for noise-free RMSE. These are designed mathematical functions with biological analogies, not measured biology.

## UCI Yeast protein localization

- Source: https://archive.ics.uci.edu/dataset/110/yeast
- Citation: Nakai, K. (1991). Yeast [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5KG68
- License: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/ (UCI dataset page). Raw data and original `.names` documentation are retained unchanged in `data/raw/`.
- 1,484 proteins, 8 numeric features and 10 localization classes. Protein identifier is excluded from the predictors. Read original `.names` for feature descriptions.
- Task: predict cellular localization from sequence-derived features; primary metric macro-F1, secondary balanced accuracy, accuracy and log loss.
- Stratified random 60/20/20 splits, saved protein identifiers; training-only standardization. Rare classes are preserved, including those with only a handful of proteins.
- Limitation: random splits are not homology-aware. No accessions-to-sequence homology clustering is implemented. This is internal validation, not novel-protein-family validation.
- “Protein localization” is a categorical destination label; it is not the localized scalar-response phenomenon modeled by the bump target.

## Wisconsin Diagnostic Breast Cancer

- Source: https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic
- Original work: Street WH, Wolberg WH, Mangasarian OL. Nuclear feature extraction for breast tumor diagnosis (1993); see source dataset page.
- Loaded through `sklearn.datasets.load_breast_cancer`, which distributes a copy of the UCI data. No external download is required. Exact input-array hash is recorded in `data/manifest.json`.
- 569 cases, 30 numeric cell-nucleus morphology features; labels malignant/benign. IDs and labels are excluded from features.
- Stratified random 60/20/20 splits with training-only standardization; selected by validation cross-entropy; report macro-F1, balanced accuracy, accuracy and log loss.
- Limitation: a small historical dataset with no independent-site external validation here. It is a biological classification benchmark, not a periodic or spatial-response experiment.

## Proposed periodic and localized extensions (not run)

- **Periodic**: Spellman et al. yeast cell-cycle data, or NCBI GEO GSE54650 circadian expression. Predict preselected genes from time and measured conditions. Hold out whole time blocks; separate interpolation from future-time extrapolation. Keep replicate/animal groups intact. Select genes and preprocessing using training data only. Include a cosinor/harmonic-regression baseline. The legacy Stanford Spellman download returned HTTP 503 during setup; no replacement data are silently substituted.
- **Localized**: replicated concentration × time expression/viability measurements. Compare models on held-out dose/time regions, include a Hill-curve baseline, and retain replicate or experimental-plate grouping. For spatial transcriptomics, use tissue-coordinate blocks and donor-level evaluation. The present repository does not contain these datasets or claim their results.

These extensions need explicit metadata checks before execution. Do not repurpose random protein/case splits as evidence of extrapolation on time or space.
