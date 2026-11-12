# spatial_benchmark — robust, uncertainty-aware benchmarking of spatial biomarker (SVG) identification

Branch `spatial-svg-benchmark`.

**Start with [FRAMEWORK.md](FRAMEWORK.md)** (also `FRAMEWORK.html`, self-contained): the combined framework (v1 + v2), what is new and why, held-out validation with a full ablation, how uncertainty is quantified and whether it makes biological sense, and how to run on real data (DLPFC). Run it on Colab: [notebooks/colab_spatial_svg_benchmark.ipynb](https://colab.research.google.com/github/wpuzhu/BIOINF-590-Individual-Project/blob/spatial-svg-benchmark/spatial_benchmark/notebooks/colab_spatial_svg_benchmark.ipynb).

History: [REPORT_v2.md](REPORT_v2.md) (v2) and [PROPOSAL.md](PROPOSAL.md) (v1): the ideas, the experimental design and the pilot results, in one file. `PROPOSAL.html` is the same document as a single self-contained page (figures and equations embedded) for reading offline.

| Path | What it contains |
|---|---|
| `stbench/simulate.py` | Simulator with known domains and SVGs: Visium-like hex grid, layered or blob domains, soft boundaries, NB counts, non-spatial cell states. Also perturbation operators (depth thinning, spot dropout, coordinate jitter) |
| `stbench/methods.py` | Domain callers (expression-only and spatial). Global SVG tests: Moran's I, Geary's C, a SPARK-X-style kernel test, HVG. The SpaGCN domain-SVG rule. ACAT and BH |
| `stbench/ensemble.py` | Majority vote (paper baseline). Proposed co-association consensus with label-free weights and pruning, plus per-spot confidence |
| `stbench/svg_aggregate.py` | Intersection reference (paper) vs ACAT and rank aggregation. Agreement score. Stability selection with the Meinshausen–Bühlmann bound |
| `scripts/run_pilot.py` | Pilot experiments E1–E5 |
| `scripts/make_figures.py` | Figures and `results/pilot/summary.json` |
| `stbench/splitting.py`, `stbench/domain_svg.py`, `stbench/workflow.py` | v2: count splitting, NB offset test, spatial gate, negative controls, SplitStab |
| `stbench/real/` | v2: DLPFC loaders, GPU method wrappers (GraphST, STAGATE, SpaGCN), real-data experiments R1–R6 |
| `scripts/run_validation.py`, `scripts/make_validation_figures.py` | v2 held-out validation V1–V5 (`results/validation/`) |
| `scripts/run_combined.py`, `scripts/make_combined_figures.py` | Combined-framework validation and ablation (`results/combined/`) |
| `scripts/run_real.py`, `scripts/report_real.py`, `configs/` | v2 real-data driver and HTML report |
| `notebooks/colab_spatial_svg_benchmark.ipynb` | v2 Colab runner |
| `tests/` | Unit tests, including null calibration of every test |

```bash
cd spatial_benchmark
python -m unittest discover -s tests          # ~1 s
python scripts/run_pilot.py --out results/pilot --workers 2   # ~25 min on 2 cores
python scripts/make_figures.py --pilot results/pilot
python scripts/run_validation.py --out results/validation --workers 2   # v2 validation, ~1-2 h on 2 cores
python scripts/make_validation_figures.py results/validation
python scripts/run_combined.py --out results/combined --workers 2 --B 20   # combined framework, ~30 min on 2 cores
python scripts/make_combined_figures.py results/combined
python scripts/run_real.py --config configs/real_quick.json             # real data (needs anndata, scanpy, huggingface_hub)
```

Dependencies: numpy, scipy, scikit-learn, pandas, matplotlib. No GPU and no R are needed.
