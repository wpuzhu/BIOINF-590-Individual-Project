# KAN versus MLP for biological response structure: focused literature review

Puzhu (Chris) Wang · BIOINF 590 Individual Project · Working research document

## Research question

At comparable trainable-parameter budgets, how do target structure, basis family and depth affect predictive error and training cost? The project separates controlled function recovery from real biological prediction. A model winning on a synthetic bump has not thereby demonstrated a benefit for spatial transcriptomics or drug response.

## Architectural foundations

An MLP layer computes $h^{l+1}=\sigma(W_lh^l+b_l)$, with a linear output layer for regression or logits for classification. A KAN layer computes $h_j^{l+1}=b_j+\sum_i\phi_{ji}^l(h_i^l)$. Here every edge, including the final output edges, uses

$$\phi(u)=a\,\mathrm{SiLU}(u)+s\sum_{r=1}^{R}c_rB_r(u),\qquad \mathrm{SiLU}(u)=u/(1+e^{-u}).$$

Liu et al. introduce spline-based learned edge functions and describe grid adaptation and deeper networks [1]. The present implementation retains the SiLU-plus-basis structure but uses fixed grids, dense computation and no symbolic pruning or adaptive grid refinement. It is an experimental implementation, not a reproduction of all pykan results.

The Kolmogorov–Arnold representation theorem motivates the composition of univariate functions and addition. Its $2d+1$ reference width does not imply that finite smooth bases of that width fit every target, nor dictate the width of each layer in a practical deep KAN. Width and basis resolution remain hyperparameters.

Interactions require a nonlinearity after mixing. In $\psi(\phi_1(x_1)+\phi_2(x_2))$, a nonlinear $\psi$ can generate interactions. The additive intermediate sum alone cannot. One hidden KAN layer means two edge layers; it does not have the same allocation of nonlinear operations as one hidden MLP layer. Depth is therefore varied explicitly, with parameter budgets held approximately fixed.

## Why these three basis families?

| Family | Implemented basis | Structural preference | Literature and qualification |
|---|---|---|---|
| Cubic spline | Uniform cardinal cubic $B_3((u-\mu_r)/h)$ | Compact support; smooth local changes | Original KAN [1]; genomic KAN work [4]. Fixed grids here differ from adaptive grids in the original method. |
| Fourier | $\sin(k\pi u),\cos(k\pi u)$, $k=1,\ldots,R/2$ | Global oscillations; period two for the basis component | FourierKAN-GCF [3] supplies a KAN precedent, in recommendation rather than biology. The SiLU component means the complete edge is not necessarily periodic. |
| Gaussian RBF | $\exp[-(u-\mu_r)^2/(2h^2)]$ | Local responses with decaying tails | FastKAN [2] establishes an RBF alternative to cubic splines. Its normalization and engineering differ; no FastKAN speed claim transfers automatically to this code. |

All families have the same number of basis coefficients and independent learned base and expansion scales. Their costs still differ: parameter count does not equal operation count. Fixed centers cover [-1,1] even in hidden layers; hidden activations and standardized biological inputs may leave this range. Outside the basis region the SiLU path can dominate. This boundary policy must be considered when interpreting extrapolation.

## Direct biological KAN evidence

Cherednichenko and Poptsova evaluate dense LKAN and convolutional CKAN replacements on genomic sequence classification and generation [4]. Their results motivate testing architecture and depth, rather than treating KAN as uniformly better. “Linear KAN” in that work is a dense layer designation, not a network of linear scalar edge functions. Our dense tabular pipeline does not reproduce their sequence experiments.

CoxKAN uses KAN in proportional-hazards survival modeling [5]. It supports the relevance of learned univariate responses to biological outcomes. Survival models require censoring-aware objectives and metrics; the classification/regression pipeline in this repository does not implement survival analysis.

These papers do not establish that a Fourier or RBF basis will outperform splines for periodic or localized biological responses. Those are testable hypotheses below.

## From the six targets to biological questions

| Synthetic target | Isolated property | Related biological task | What would make a valid real-data experiment? |
|---|---|---|---|
| $0.8x_1-0.5x_2$ | Affine response | Approximately linear log-expression or log-risk contributions | Include ridge/logistic regression; do not infer a mechanistic linear law from predictive fit. |
| $\sin(\pi x_1)+0.5x_2^2$ | Smooth separable effects | Expression response to separate environmental or dosage covariates | Compare additive models with interaction-capable models on matched data. |
| $1.5x_1x_2$ | Non-additive interaction | Epistasis or two-drug interaction | Hold out genotype groups or drug-pair contexts; compare with additive baselines. |
| $\sin(\pi x_1)\cos(\pi x_2)$ | Periodicity plus interaction | Cell-cycle/circadian expression conditional on tissue or condition | Use recorded time, retain time blocks, and split animals/replicates as units. Fourier on an arbitrary embedding is not biological periodicity. |
| $\exp[-5((x_1-.25)^2+(x_2+.2)^2)]$ | Local joint response | Transient induction, dose windows, or expression near tissue niches | Hold out concentration regions or spatial blocks; a univariate RBF edge is not itself a spatial kernel. |
| $\max(0,x_1+x_2)-.4x_1$ | Oblique threshold with a kink | Cooperative regulatory threshold or switch-like response | Compare ReLU and smooth families; real biology may be smooth even if the toy target has a corner. |

Spellman et al.'s yeast cell-cycle expression study [6] and the mouse circadian GEO series GSE54650 [7] are suitable periodic follow-ups. For GSE54650, select genes by a fixed list or training-only criterion; perform leave-time-block-out interpolation and late-time extrapolation separately; avoid selecting rhythmic genes on the full dataset. The current UCI Yeast dataset concerns protein localization and is **not** that cell-cycle expression dataset.

For localized responses, a dose-response study with replicate-level metadata is a stronger next experiment than calling a protein-localization label a localized response. Use log concentration and exposure time as continuous inputs, hold out whole concentration/time regions, and include a Hill-curve baseline where appropriate. Spatial transcriptomics is another candidate, but spatial blocks and donor-level validation are necessary to avoid nearby spots leaking information across splits. These follow-ups are proposed, not completed experiments.

## Experimental interpretation

Validation-based stopping measures an optimization outcome under a particular optimizer, initialization and learning-rate search. It does not reveal full representational capacity. The primary comparison reports both attained test performance and cost. Train-to-plateau time is meaningful alongside the stopping status; an epoch-limit run is not evidence of convergence. Both the selected fit time and the full learning-rate-search cost are saved.

Inside-range test points and outside-range test points are independently sampled, never used for stopping. Biological datasets use stratified random splits with training-only standardization; these establish internal held-out performance, not external cohort or homology-aware validation. Repeated split seeds are dependent resamples of one dataset, so their standard deviations and bootstrap intervals are descriptive, not population-level confidence guarantees.

## References

1. Liu Z et al. **KAN: Kolmogorov–Arnold Networks.** arXiv:2404.19756 (2024; revised 2025). https://arxiv.org/abs/2404.19756
2. Li Z. **Kolmogorov-Arnold Networks are Radial Basis Function Networks.** arXiv:2405.06721 (2024). https://arxiv.org/abs/2405.06721
3. Xu J et al. **Enhancing Graph Collaborative Filtering with FourierKAN Feature Transformation.** arXiv:2406.01034 (2024; revised 2025). https://arxiv.org/abs/2406.01034
4. Cherednichenko O, Poptsova M. **Kolmogorov–Arnold networks for genomic tasks.** Briefings in Bioinformatics 26(2), bbaf129 (2025). https://doi.org/10.1093/bib/bbaf129
5. **CoxKAN: Kolmogorov-Arnold networks for interpretable, high-performance survival analysis.** Bioinformatics 41(8), btaf413 (2025). https://doi.org/10.1093/bioinformatics/btaf413
6. Spellman PT et al. **Comprehensive identification of cell cycle-regulated genes of the yeast Saccharomyces cerevisiae by microarray hybridization.** Molecular Biology of the Cell 9, 3273–3297 (1998). https://doi.org/10.1091/mbc.9.12.3273
7. NCBI GEO. **GSE54650: A circadian gene expression atlas in mammals assayed by microarray.** https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE54650
8. Nakai K. **Yeast [Dataset].** UCI Machine Learning Repository. https://doi.org/10.24432/C5KG68
9. Wolberg W, Mangasarian O, Street N, Street W. **Breast Cancer Wisconsin (Diagnostic) [Dataset].** UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic
