# BIOINF 590 · Spatially variable gene detection

Can we combine spatially variable gene (SVG) callers to recover more true genes at a comparable false-discovery rate, while showing which calls survive changes in depth, spatial sampling and method choice?

- [Literature review](docs/literature.md) covers SVG definitions, method families, benchmarks, published ensembles and the remaining research gap.
- [Methods and benchmark protocol](docs/benchmark.md) specifies the ensemble, uncertainty outputs, baselines, experimental units and decision criteria.
- [Dataset and software catalog](docs/resources.md) lists acquisition sources, inputs, evidence and implementation status.
- [Novelty audit and revised proposal](docs/novelty.md) identifies what is established, what remains testable, and how to adapt the design to other SVG definitions.
- [Expanded real-data verification](spatial_benchmark/results/transfer_predictions/README.md) adds Visium, seqFISH and Slide-seqV2 results and compares the ensemble with simpler scores.
- [Direct real-data predictions](docs/prediction.md) explains the fixed inference workflow and links to the ranked gene tables from three Visium and two MERFISH sections.
- [Existing domain-marker pilot](spatial_benchmark/FRAMEWORK.md) summarizes the branch's completed simulations. [Reproduction commands](spatial_benchmark/README.md) cover the simulator, tests and DLPFC runner.

The primary study targets **overall SVGs**. A secondary study asks how uncertainty in inferred tissue domains affects **domain-marker SVGs**. The official-caller ensemble and the cross-dataset benchmark are proposed. The repository includes measured-section predictions from the fixed prototype and the previous simulation outputs.
