# Existing domain-marker pilot

This report summarizes the completed simulation work that informs the [new SVG benchmark](../docs/benchmark.md). The pilot combines count splitting, weighted spatial-domain consensus, depth-aware marker tests, a spatial gate and perturbation-based selection. Published workflows were approximated with CPU stand-ins, including a SPARK-X-style kernel test.

## Procedure

Each replicate subsamples spots, appends negative-control genes generated with the slice's depth profile and splits counts using a beta-binomial rule. Domain callers use the discovery half. The pipeline scores their partitions by replicate agreement, centrality and spatial coherence, prunes the panel and builds weighted co-association consensus. Marker tests on the other half compare depth-adjusted rates in domain core spots. A combined Moran and kernel statistic gates candidates for spatial evidence. Splitting addresses count reuse under the specified model and the spatial gate targets nonspatial states that yield real expression differences between inferred clusters.

Across replicates the pipeline summarizes spot agreement, gene selection frequency and modal domain. Its threshold uses an FDP proxy derived from selected synthetic controls. That proxy depends on how well generated controls represent technical false calls. The recorded Meinshausen–Bühlmann calculation requires assumptions beyond the perturbation procedure and should be treated as a diagnostic. Perturbation frequencies quantify repeatability under the experiment's resampling scheme.

## Simulation results

Six signal tissues (layered and blob-shaped) and four all-null tissues were evaluated. The table summarizes the recorded results with 20 perturbation replicates in the combined procedure. [Raw summary](results/combined/summary.json), [ablation](results/combined/ablation.csv), [figure](results/combined/figures/c1_gene_level.png).

| Pipeline | Mean all-null calls | Full-depth precision / recall | 25% depth precision / recall |
|---|---:|---:|---:|
| Kang-style strict | 22.8 | 1.00 / 0.26 | 1.00 / 0.02 |
| Kang-style liberal | 107.8 | 0.90 / 0.85 | 0.73 / 0.48 |
| Same-data stability | 79.2 | 1.00 / 0.80 | 0.99 / 0.41 |
| Single split and gate | 0 | 1.00 / 0.55 | 1.00 / 0.32 |
| Combined | 0 | 1.00 / 0.58 | 1.00 / 0.30 |

The all-null arm exposed reproducible false calls in the same-data stability pipeline. A single split and gate performed similarly to the combined procedure, making it an essential comparator. The pilot had limited independent tissues, and the branch's previous validation found modest and uncertain gains from pruning relative to majority vote. The recorded precision and null counts characterize this simulator configuration. [Validation data](results/validation/summary.json), [spot uncertainty](results/combined/figures/c2_spot_uncertainty.png), [gene uncertainty](results/combined/figures/c3_gene_uncertainty.png).

## What can be reproduced

The simulator, earlier pilot, validation runs, combined runs and DLPFC runner are documented in the [run guide](README.md). The simulated pipeline also produced a [demo report](results/sim_pipeline_demo/report.html). Observed results are from simulations. Real-data and GPU-wrapper results remain pending.
