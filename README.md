# BIOINF 590 · Spatially variable gene detection

Can we combine spatially variable gene (SVG) callers to recover more true genes at a comparable false-discovery rate, while showing which calls survive changes in depth, spatial sampling and method choice?

- [Literature review](docs/literature.md) covers SVG definitions, method families, benchmarks, published ensembles and the remaining research gap.
- [Methods and benchmark protocol](docs/benchmark.md) specifies the ensemble, uncertainty outputs, baselines, experimental units and decision criteria.
- [Dataset and software catalog](docs/resources.md) lists acquisition sources, inputs, evidence and implementation status.
- [Existing domain-marker pilot](spatial_benchmark/FRAMEWORK.md) summarizes the branch's completed simulations. [Reproduction commands](spatial_benchmark/README.md) cover the simulator, tests and DLPFC runner.

The primary study targets **overall SVGs**. A secondary study asks how uncertainty in inferred tissue domains affects **domain-marker SVGs**. The official-caller ensemble and the cross-dataset benchmark are proposed. The repository includes the previous simulation outputs and a real-data runner.
