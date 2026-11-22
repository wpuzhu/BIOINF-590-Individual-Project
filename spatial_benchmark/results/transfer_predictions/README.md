# Expanded real-data verification

The fixed 75/25 rank ensemble was applied to three additional measured inputs without tuning. The original five predictions were reused for the ablation. The two larger inputs were scored on seeded subsets of 8,000 locations. Stability was then evaluated by 20 draws of 80% of those working locations.

| Input | Scored / available locations | Genes | Top candidates with selection frequency ≥0.8 | Table |
|---|---:|---:|---:|---|
| DLPFC Visium 151675 | 3,592 / 3,592 | 2,000 | 81 / 100 | [CSV](dlpfc_151675.csv) |
| Lohoff mouse embryo seqFISH | 8,000 / 19,416 | 351 | 15 / 20 | [CSV](seqfish_mouse_embryo.csv) |
| Stickels mouse hippocampus Slide-seqV2 | 8,000 / 41,786 | 604 | 86 / 100 | [CSV](slideseqv2_mouse_hippocampus.csv) |

The seqFISH input was count-normalized. Slide-seqV2 used the published log-normalized representation as supplied. These platform runs establish execution and within-input stability. Each has one input for this study, so biological replication and cross-platform accuracy are unmeasured.

## Matched top-list overlap

For each pair the table uses only shared eligible genes. The ensemble column uses saved full-precision ranks. The component and ablation scores were derived from exported caller scores, which were rounded. Ties are ordered by gene ID. Pair overlaps sharing a section are dependent.

| Pair | List size | Moran’s I | Equal-family mean | 75/25 ensemble |
|---|---:|---:|---:|---:|
| Visium 151673 / 151674 | 100 | 84 | 72 | 76 |
| Visium 151673 / 151507 | 100 | 67 | 60 | 59 |
| Visium 151674 / 151507 | 100 | 66 | 59 | 59 |
| Visium 151673 / 151675 | 100 | 91 | 70 | 70 |
| Visium 151674 / 151675 | 100 | 83 | 70 | 71 |
| Visium 151507 / 151675 | 100 | 70 | 61 | 65 |
| MERFISH, same animal | 10 | 9 | 8 | 8 |

Moran's I exceeds the ensemble on this overlap metric in all seven comparisons. The [full ablation](rank_ablation.csv) also includes Geary’s C, kernel-only, flat caller mean, family maximum and Spearman correlation. Concordance is an agreement endpoint. Test accuracy and false-discovery control with known truth before making superiority claims. The [score diagnostics](score_diagnostics.csv) include mitochondrial-symbol counts and distinct exported kernel scores. Its stability count refers to the **ensemble’s** resampled selections among each method's top genes.

[Input checksums and versions](manifest.json) · [Original real predictions](../real_predictions/README.md) · [Novelty audit and revised method](../../../docs/novelty.md)

To reproduce, use the [direct scoring command](../../../docs/prediction.md) with [transfer configuration](../../configs/predict_transfer.json), then generate the descriptive comparison:

```bash
cd spatial_benchmark
python scripts/audit_svg_rankings.py --runs results/real_predictions results/transfer_predictions --out results/transfer_predictions
```
