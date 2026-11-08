### Headline: validation-selected KAN vs MLP per task (pooled over depth and budget)

ratio = exp(mean log(KAN error / MLP error)); < 1 means KAN better. CI: bootstrap over replicate seeds. Baselines: RMSE of predicting the training mean / of ridge regression (cosinor for circadian tasks).

| Task | Audited structure | Inside ratio (95% CI) | KAN wins | KAN / MLP inside error | Baseline mean / ridge | Outside ratio (95% CI) | KAN wins | KAN / MLP outside error | Baseline mean / ridge |
|---|---|---|---:|---|---|---|---:|---|---|
| circadian_blood_aligned |  | 0.998 (0.995–1.003) | 11/20 | 1.011 / 1.013 | 1.092 / 1.061 (cosinor 0.999) | — |  |  |  |
| circadian_liver_aligned |  | 1.033 (0.999–1.060) | 4/20 | 0.552 / 0.535 | 1.142 / 1.125 (cosinor 0.487) | 0.443 (0.391–0.510) | 20/20 | 1.519 / 3.526 | 1.258 / 1.256 (cosinor 0.663) |

### Validation-selected KAN vs MLP, inside test

| Task | Depth | Budget | n | KAN error | MLP error | KAN − MLP (95% CI) | KAN wins | Holm p | Usual picks |
|---|---:|---|---:|---:|---:|---|---:|---:|---|
| circadian_blood_aligned | 1 | large | 5 | 1.007 | 1.012 | -0.005 (-0.029, 0.018) | 3/5 | 1.000 | kan-rbf / mlp-relu |
| circadian_blood_aligned | 1 | small | 5 | 1.013 | 1.013 | -0.000 (-0.016, 0.015) | 3/5 | 1.000 | kan-fourier / mlp-relu |
| circadian_blood_aligned | 2 | large | 5 | 1.011 | 1.010 | +0.002 (-0.004, 0.008) | 2/5 | 1.000 | kan-rbf / mlp-silu |
| circadian_blood_aligned | 2 | small | 5 | 1.013 | 1.016 | -0.003 (-0.011, 0.005) | 3/5 | 1.000 | kan-fourier / mlp-silu |
| circadian_liver_aligned | 1 | large | 5 | 0.546 | 0.524 | +0.022 (-0.008, 0.053) | 1/5 | 1.000 | kan-rbf / mlp-relu |
| circadian_liver_aligned | 1 | small | 5 | 0.538 | 0.522 | +0.016 (-0.008, 0.041) | 1/5 | 1.000 | kan-rbf / mlp-relu |
| circadian_liver_aligned | 2 | large | 5 | 0.563 | 0.553 | +0.010 (-0.041, 0.060) | 1/5 | 1.000 | kan-spline / mlp-silu |
| circadian_liver_aligned | 2 | small | 5 | 0.559 | 0.540 | +0.020 (-0.011, 0.051) | 1/5 | 1.000 | kan-spline / mlp-silu |

### Validation-selected KAN vs MLP, outside (extrapolation) test

| Task | Depth | Budget | n | KAN error | MLP error | KAN − MLP (95% CI) | KAN wins | Holm p |
|---|---:|---|---:|---:|---:|---|---:|---:|
| circadian_liver_aligned | 1 | large | 5 | 1.682 | 3.394 | -1.711 (-2.182, -1.240) | 5/5 | 0.250 |
| circadian_liver_aligned | 1 | small | 5 | 1.440 | 3.409 | -1.969 (-3.032, -0.907) | 5/5 | 0.250 |
| circadian_liver_aligned | 2 | large | 5 | 1.498 | 3.655 | -2.157 (-3.680, -0.634) | 5/5 | 0.250 |
| circadian_liver_aligned | 2 | small | 5 | 1.454 | 3.646 | -2.192 (-3.615, -0.768) | 5/5 | 0.250 |

### Six-model mean error by task (inside test; mean over seeds, depths and budgets)

| Task | mlp-tanh | mlp-relu | mlp-silu | kan-spline | kan-fourier | kan-rbf |
|---|---:|---:|---:|---:|---:|---:|
| circadian_blood_aligned | 1.012 | 1.013 | 1.014 | 1.012 | 1.016 | **1.011** |
| circadian_liver_aligned | 0.536 | **0.526** | 0.567 | 0.546 | 0.692 | 0.543 |

Friedman test over 8 blocks: chi² = 14.1, p = 0.015; Nemenyi critical difference = 2.67 ranks.

| Model | Mean rank |
|---|---:|
| mlp-tanh | 2.62 |
| mlp-relu | 2.75 |
| kan-rbf | 2.75 |
| kan-spline | 3.38 |
| mlp-silu | 4.00 |
| kan-fourier | 5.50 |
