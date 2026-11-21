# Direct predictions on real spatial transcriptomics sections

Three human DLPFC Visium sections and two mouse hypothalamus MERFISH sections were scored with the [fixed inference workflow](../../../docs/prediction.md). Each file includes every screened gene, individual caller scores, ensemble rank, perturbation rank quantiles and top-k frequency.

| Section | Locations | Genes scored | Candidate table |
|---|---:|---:|---|
| DLPFC 151673 | 3,639 | 2,000 | [CSV](dlpfc_151673.csv) |
| DLPFC 151674 | 3,673 | 2,000 | [CSV](dlpfc_151674.csv) |
| DLPFC 151507 | 4,226 | 2,000 | [CSV](dlpfc_151507.csv) |
| MERFISH, animal 1, Bregma −14 | 6,605 | 160 | [CSV](merfish_bregma_m14.csv) |
| MERFISH, animal 1, Bregma −19 | 6,507 | 160 | [CSV](merfish_bregma_m19.csv) |

The adjacent DLPFC sections 151673 and 151674 shared **76 of 100 candidates** after restricting to commonly tested genes. Each comparison with 151507 shared **59 of 100**. The MERFISH sections shared **8 of 10**, including **Nnat**, which ranked first in both and in every location subsample. The MERFISH result measures within-animal agreement. These overlaps describe concordance of candidate lists.

[Section agreement CSV](section_overlap.csv) · [Input sources, checksums and software versions](manifest.json)
