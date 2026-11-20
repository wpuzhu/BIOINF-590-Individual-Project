# Dataset and software catalog

## Datasets

Acquire raw or minimally processed counts, coordinates and sample metadata. Record the release, checksum, license, biological unit and any transformations before making analysis inputs. The existing simulator and its outputs are in the branch. External data below are cataloged by source.

| Priority | Resource | What to acquire | Benchmark role |
|---|---|---|---|
| Start | [Repository NB simulator](../spatial_benchmark/stbench/simulate.py) | Generate Visium-like grids, layers/blobs, nonspatial states and truth labels. Effects, seed and perturbation must be recorded. | Smoke test and ablation on known genes and domains. Generate fresh seeds for the new study. |
| Start | [Patel benchmark](https://github.com/pinellolab/SVG_Benchmarking), [Open Problems SVG task](https://openproblems.bio/benchmarks/spatially_variable_genes/) | Use benchmark input and simulator release pointers, method runners and dataset metadata. Keep any precomputed results as external comparisons. | Independent ranking, calibration and resource comparisons. Confirm exact generator parameters and truth definitions. |
| Start | [Maynard DLPFC, spatialLIBD](https://bioconductor.org/packages/release/data/experiment/html/spatialLIBD.html) | Retrieve the spatialLIBD experiment with counts, spot coordinates, sample/donor IDs and manual layer annotations. Use 151673 for ingestion checks, then account for all 12 sections from three donors. | Replication by adjacent pair and donor, semi-synthetic geometry, annotated-domain and marker recovery. |
| Start | [scDesign3](https://bioconductor.org/packages/release/bioc/html/scDesign3.html) and [SRTsim](https://xzhoulab.github.io/SRTsim/) | Fit profiles using development slices. Save masks, size factors, embedded patterns and gene truth. SRTsim supplies a second model for the sensitivity arm. | Truth-known simulations with realistic count variation and geometry. |
| Transfer | [Moffitt et al. hypothalamic MERFISH](https://doi.org/10.5061/dryad.8t8s248) | Download the all-cell CSV. Select sections by animal and imaging metadata, retain cell coordinates and measured panel, and avoid combining sections in a single coordinate frame. | Targeted imaging panel, smaller spatial scales, and stability across sections or animals where comparable. |
| Transfer | [Stickels et al. Slide-seqV2, SCP815](https://singlecell.broadinstitute.org/single_cell/study/SCP815) | Select a hippocampal puck with barcoded counts and bead coordinates. Record puck ID and preprocessing. | High-resolution sequencing, runtime and depth robustness. Cross-puck replication requires matched independent pucks. |
| Extension | [seqFISH+](https://github.com/CaiGroup/seqFISH-PLUS) and [SVG atlas preprint](https://arxiv.org/abs/2510.07653) | Select a documented mouse brain field/section or a released atlas subset with original study and donor IDs. | External platform/tissue sensitivity. Check overlap with development studies before labeling an arm external. |

## Methods to run

| Method | Source and role | Inputs and outputs to retain |
|---|---|---|
| SPARK-X and SPARK | [Official SPARK R repository](https://github.com/xzhoulab/SPARK). SPARK-X is a required scalable baseline. SPARK is optional due to cost. | Gene-by-location counts and matching coordinates. Record raw combined p-values, filters and optional covariates. |
| Moran's I and Geary's C | [Squidpy spatial autocorrelation API](https://squidpy.readthedocs.io/en/stable/api/squidpy.gr.spatial_autocorr.html). Required Moran baseline and optional related caller. | AnnData with normalized expression and spatial graph. Fix graph and direction, record statistic and unadjusted inference output. |
| nnSVG | [Bioconductor nnSVG](https://bioconductor.org/packages/release/bioc/html/nnSVG.html). Required GP comparator. | SpatialExperiment and normalized expression. Retain LRT, raw p-value, spatial variance and length-scale when returned. |
| Spanve | [Official Python repository](https://github.com/zjupgx/Spanve). Required local-distribution comparator. | Use package-recommended expression and coordinates. Record statistic, raw p-value if available and graph settings. |
| SpatialDE/SpatialDE2 and SpaGFT | [SpatialDE](https://github.com/Teichlab/SpatialDE), [SpatialDE2](https://github.com/PMBio/SpatialDE), [SpaGFT benchmark wrapper](https://github.com/pinellolab/SVG_Benchmarking). Secondary coverage of GP and graph spectral scores. | Preserve method-specific normalization and score direction. Use rank output for score-only modes. |
| Castl | [Author code](https://github.com/TheY11/Castl) and [archived release](https://zenodo.org/records/16751799). Primary published ensemble comparator. | Save module, member panel, cutoffs and whether the run used author artifacts or a reproduction. The local Castl-style functions are prototypes. |
| Domain-markers | [SpaGCN](https://github.com/jianhuupenn/SpaGCN), [DESpace](https://bioconductor.org/packages/release/bioc/html/DESpace.html), [existing pipeline](../spatial_benchmark/stbench/pipeline.py). | Keep domain definitions, tested contrasts, effect direction, multiplicity family and confidence or assignment stability. |

Cross-language exchange should preserve sparse integer counts, stable gene and location IDs, x/y units, section IDs, sample grouping, and layer or marker labels where present. Standardize outputs to section, gene, method, score, raw p-value, effect if defined, run/seed, input gene universe and status. Check spot and gene order after conversion between AnnData and SpatialExperiment.

Official detector wrappers and external acquisitions are planned for the new benchmark. The branch contains CPU stand-ins plus a DLPFC runner and optional domain callers. The latter's mirror input should be checked against spatialLIBD before benchmark use.
