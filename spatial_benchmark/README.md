# spatial_benchmark — robust, uncertainty-aware benchmarking of spatial biomarker (SVG) identification

Branch `spatial-svg-benchmark`. **Start with [PROPOSAL.md](PROPOSAL.md)**: the ideas, the experimental design and the pilot results, in one file. `PROPOSAL.html` is the same document as a single self-contained page (figures and equations embedded) for reading offline.

| Path | What it contains |
|---|---|
| `stbench/simulate.py` | Simulator with known domains and SVGs: Visium-like hex grid, layered or blob domains, soft boundaries, NB counts, non-spatial cell states. Also perturbation operators (depth thinning, spot dropout, coordinate jitter) |
| `stbench/methods.py` | Domain callers (expression-only and spatial). Global SVG tests: Moran's I, Geary's C, a SPARK-X-style kernel test, HVG. The SpaGCN domain-SVG rule. ACAT and BH |
| `stbench/ensemble.py` | Majority vote (paper baseline). Proposed co-association consensus with label-free weights and pruning, plus per-spot confidence |
| `stbench/svg_aggregate.py` | Intersection reference (paper) vs ACAT and rank aggregation. Agreement score. Stability selection with the Meinshausen–Bühlmann bound |
| `scripts/run_pilot.py` | Pilot experiments E1–E5 |
| `scripts/make_figures.py` | Figures and `results/pilot/summary.json` |
| `tests/` | Unit tests, including null calibration of every test |

```bash
cd spatial_benchmark
python -m unittest discover -s tests          # ~1 s
python scripts/run_pilot.py --out results/pilot --workers 2   # ~25 min on 2 cores
python scripts/make_figures.py --pilot results/pilot
```

Dependencies: numpy, scipy, scikit-learn, pandas, matplotlib. No GPU and no R are needed.
