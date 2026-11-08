### Headline: validation-selected KAN vs MLP per task (pooled over depth and budget)

ratio = exp(mean log(KAN error / MLP error)); < 1 means KAN better. CI: bootstrap over replicate seeds. Baselines: RMSE of predicting the training mean / of ridge regression (cosinor for circadian tasks).

| Task | Audited structure | Inside ratio (95% CI) | KAN wins | KAN / MLP inside error | Baseline mean / ridge | Outside ratio (95% CI) | KAN wins | KAN / MLP outside error | Baseline mean / ridge |
|---|---|---|---:|---|---|---|---:|---|---|
| linear | near-linear | 1.924 (1.610–2.298) | 0.0/20.0 | 0.010 / 0.005 | 0.551 / 0.002 | 14.992 (11.498–19.548) | 0.0/20.0 | 0.222 / 0.018 | 0.976 / 0.003 |
| additive | nonlinear marginals; oscillatory marginal | 1.214 (1.050–1.384) | 4.0/20.0 | 0.013 / 0.011 | 0.747 / 0.469 | 1.893 (1.529–2.316) | 0.0/20.0 | 0.498 / 0.273 | 0.966 / 1.546 |
| interaction | interaction | 1.532 (1.409–1.648) | 0.0/20.0 | 0.014 / 0.009 | 0.504 / 0.504 | 6.578 (6.011–7.199) | 0.0/20.0 | 0.624 / 0.096 | 1.437 / 1.437 |
| waves | interaction | 0.652 (0.564–0.732) | 20.0/20.0 | 0.015 / 0.023 | 0.502 / 0.502 | 1.162 (0.994–1.345) | 5.0/20.0 | 0.382 / 0.330 | 0.504 / 0.505 |
| radial | nonlinear marginals; interaction; single-extremum (bump/U) marginal | 0.697 (0.644–0.780) | 19.0/20.0 | 0.010 / 0.015 | 0.279 / 0.218 | 0.748 (0.625–0.873) | 15.0/20.0 | 0.068 / 0.087 | 0.006 / 0.213 |
| kink | nonlinear marginals; interaction; threshold/saturation | 1.499 (1.377–1.632) | 1.0/20.0 | 0.014 / 0.009 | 0.499 / 0.236 | 5.935 (5.015–7.130) | 0.0/20.0 | 0.199 / 0.034 | 0.914 / 0.479 |
| breast_cancer | weak interaction; single-extremum (bump/U) marginal; threshold/saturation | 1.471 (0.901–2.211) | 4.0/20.0 | 0.040 / 0.027 |  | — |  |  |  |
| circadian_blood | nonlinear marginals; single-extremum (bump/U) marginal; 24 h rhythm in 100% of outputs (q<0.05) | 0.999 (0.997–1.001) | 9.0/20.0 | 1.011 / 1.012 | 1.092 / 1.061 (cosinor 0.999) | — |  |  |  |
| circadian_liver | nonlinear marginals; weak interaction; oscillatory marginal; 24 h rhythm in 100% of outputs (q<0.05) | 1.026 (0.978–1.071) | 7.0/20.0 | 0.538 / 0.526 | 1.142 / 1.125 (cosinor 0.487) | 0.492 (0.465–0.531) | 20.0/20.0 | 1.545 / 3.219 | 1.258 / 1.256 (cosinor 0.663) |
| diabetes | near-linear; threshold/saturation | 1.037 (0.999–1.065) | 5.0/20.0 | 0.728 / 0.702 | 0.957 / 0.696 | — |  |  |  |
| drug_combo | nonlinear marginals; interaction; single-extremum (bump/U) marginal; threshold/saturation | 0.990 (0.946–1.025) | 22.0/48.0 | 0.598 / 0.600 | 0.954 / 0.756 | 1.036 (0.928–1.143) | 21.0/48.0 | 0.875 / 0.835 | 1.668 / 1.277 |
| gb1_epistasis | weak interaction | 1.017 (1.007–1.026) | 2.0/20.0 | 0.305 / 0.300 | 0.999 / 0.360 | 1.226 (1.142–1.313) | 0.0/20.0 | 0.392 / 0.319 | 1.878 / 0.919 |
| mpsa_splicing | interaction | 1.005 (0.996–1.014) | 8.0/20.0 | 0.597 / 0.594 | 0.954 / 0.864 | — |  |  |  |
| yeast | near-linear; single-extremum (bump/U) marginal; threshold/saturation | 0.987 (0.918–1.050) | 10.0/20.0 | 0.475 / 0.480 |  | — |  |  |  |

### Validation-selected KAN vs MLP, inside test

| Task | Depth | Budget | n | KAN error | MLP error | KAN − MLP (95% CI) | KAN wins | Holm p | Usual picks |
|---|---:|---|---:|---:|---:|---|---:|---:|---|
| additive | 1 | large | 5 | 0.012 | 0.010 | +0.002 (-0.004, 0.007) | 2/5 | 1.000 | kan-rbf / mlp-silu |
| additive | 1 | small | 5 | 0.013 | 0.012 | +0.002 (-0.004, 0.007) | 2/5 | 1.000 | kan-rbf / mlp-relu |
| additive | 2 | large | 5 | 0.014 | 0.012 | +0.003 (0.001, 0.004) | 0/5 | 1.000 | kan-rbf / mlp-silu |
| additive | 2 | small | 5 | 0.014 | 0.011 | +0.003 (0.001, 0.005) | 0/5 | 1.000 | kan-rbf / mlp-silu |
| breast_cancer | 1 | large | 5 | 0.040 | 0.030 | +0.009 (-0.012, 0.031) | 2/5 | 1.000 | kan-spline / mlp-tanh |
| breast_cancer | 1 | small | 5 | 0.043 | 0.026 | +0.017 (-0.009, 0.043) | 1/5 | 1.000 | kan-spline / mlp-relu |
| breast_cancer | 2 | large | 5 | 0.040 | 0.026 | +0.013 (-0.015, 0.042) | 1/5 | 1.000 | kan-spline / mlp-silu |
| breast_cancer | 2 | small | 5 | 0.038 | 0.026 | +0.011 (-0.002, 0.024) | 0/5 | 1.000 | kan-rbf / mlp-relu |
| circadian_blood | 1 | large | 5 | 1.012 | 1.013 | -0.001 (-0.016, 0.013) | 2/5 | 1.000 | kan-rbf / mlp-relu |
| circadian_blood | 1 | small | 5 | 1.014 | 1.012 | +0.002 (-0.004, 0.008) | 1/5 | 1.000 | kan-rbf / mlp-relu |
| circadian_blood | 2 | large | 5 | 1.009 | 1.011 | -0.002 (-0.014, 0.009) | 2/5 | 1.000 | kan-rbf / mlp-relu |
| circadian_blood | 2 | small | 5 | 1.011 | 1.014 | -0.003 (-0.012, 0.006) | 4/5 | 1.000 | kan-rbf / mlp-silu |
| circadian_liver | 1 | large | 5 | 0.544 | 0.527 | +0.017 (-0.007, 0.041) | 1/5 | 1.000 | kan-rbf / mlp-relu |
| circadian_liver | 1 | small | 5 | 0.536 | 0.524 | +0.012 (-0.062, 0.085) | 1/5 | 1.000 | kan-rbf / mlp-tanh |
| circadian_liver | 2 | large | 5 | 0.537 | 0.514 | +0.023 (-0.018, 0.063) | 2/5 | 1.000 | kan-spline / mlp-relu |
| circadian_liver | 2 | small | 5 | 0.536 | 0.540 | -0.004 (-0.028, 0.021) | 3/5 | 1.000 | kan-rbf / mlp-silu |
| diabetes | 1 | large | 5 | 0.734 | 0.699 | +0.035 (-0.008, 0.077) | 1/5 | 1.000 | kan-rbf / mlp-silu |
| diabetes | 1 | small | 5 | 0.720 | 0.711 | +0.010 (-0.022, 0.041) | 2/5 | 1.000 | kan-rbf / mlp-tanh |
| diabetes | 2 | large | 5 | 0.724 | 0.701 | +0.023 (-0.016, 0.062) | 1/5 | 1.000 | kan-rbf / mlp-silu |
| diabetes | 2 | small | 5 | 0.732 | 0.698 | +0.034 (-0.035, 0.104) | 1/5 | 1.000 | kan-rbf / mlp-silu |
| drug_combo | 1 | large | 12 | 0.576 | 0.612 | -0.037 (-0.091, 0.018) | 7/12 | 1.000 | kan-rbf / mlp-tanh |
| drug_combo | 1 | small | 12 | 0.583 | 0.600 | -0.017 (-0.059, 0.025) | 6/12 | 1.000 | kan-rbf / mlp-relu |
| drug_combo | 2 | large | 12 | 0.637 | 0.593 | +0.044 (-0.019, 0.107) | 4/12 | 1.000 | kan-rbf / mlp-relu |
| drug_combo | 2 | small | 12 | 0.595 | 0.595 | +0.000 (-0.059, 0.060) | 5/12 | 1.000 | kan-rbf / mlp-relu |
| gb1_epistasis | 1 | large | 5 | 0.309 | 0.301 | +0.009 (-0.005, 0.022) | 1/5 | 1.000 | kan-spline / mlp-relu |
| gb1_epistasis | 1 | small | 5 | 0.307 | 0.300 | +0.006 (0.004, 0.008) | 0/5 | 1.000 | kan-rbf / mlp-relu |
| gb1_epistasis | 2 | large | 5 | 0.303 | 0.300 | +0.003 (-0.001, 0.006) | 1/5 | 1.000 | kan-rbf / mlp-tanh |
| gb1_epistasis | 2 | small | 5 | 0.303 | 0.300 | +0.003 (0.001, 0.005) | 0/5 | 1.000 | kan-rbf / mlp-tanh |
| interaction | 1 | large | 5 | 0.013 | 0.009 | +0.004 (0.002, 0.006) | 0/5 | 1.000 | kan-rbf / mlp-silu |
| interaction | 1 | small | 5 | 0.014 | 0.010 | +0.004 (0.002, 0.006) | 0/5 | 1.000 | kan-spline / mlp-silu |
| interaction | 2 | large | 5 | 0.015 | 0.010 | +0.005 (0.001, 0.009) | 0/5 | 1.000 | kan-spline / mlp-silu |
| interaction | 2 | small | 5 | 0.014 | 0.008 | +0.006 (0.003, 0.009) | 0/5 | 1.000 | kan-rbf / mlp-silu |
| kink | 1 | large | 5 | 0.015 | 0.009 | +0.006 (0.005, 0.007) | 0/5 | 1.000 | kan-rbf / mlp-relu |
| kink | 1 | small | 5 | 0.014 | 0.009 | +0.006 (0.004, 0.008) | 0/5 | 1.000 | kan-rbf / mlp-relu |
| kink | 2 | large | 5 | 0.013 | 0.010 | +0.002 (-0.000, 0.005) | 1/5 | 1.000 | kan-spline / mlp-relu |
| kink | 2 | small | 5 | 0.014 | 0.009 | +0.004 (0.003, 0.006) | 0/5 | 1.000 | kan-spline / mlp-relu |
| linear | 1 | large | 5 | 0.010 | 0.005 | +0.004 (0.002, 0.007) | 0/5 | 1.000 | kan-rbf / mlp-silu |
| linear | 1 | small | 5 | 0.009 | 0.005 | +0.005 (0.003, 0.006) | 0/5 | 1.000 | kan-rbf / mlp-silu |
| linear | 2 | large | 5 | 0.010 | 0.005 | +0.005 (0.002, 0.007) | 0/5 | 1.000 | kan-rbf / mlp-silu |
| linear | 2 | small | 5 | 0.009 | 0.005 | +0.004 (0.001, 0.007) | 0/5 | 1.000 | kan-rbf / mlp-silu |
| mpsa_splicing | 1 | large | 5 | 0.619 | 0.609 | +0.010 (-0.005, 0.024) | 1/5 | 1.000 | kan-rbf / mlp-relu |
| mpsa_splicing | 1 | small | 5 | 0.604 | 0.618 | -0.015 (-0.019, -0.011) | 5/5 | 1.000 | kan-rbf / mlp-tanh |
| mpsa_splicing | 2 | large | 5 | 0.574 | 0.575 | -0.001 (-0.022, 0.020) | 2/5 | 1.000 | kan-rbf / mlp-tanh |
| mpsa_splicing | 2 | small | 5 | 0.590 | 0.573 | +0.017 (0.006, 0.027) | 0/5 | 1.000 | kan-rbf / mlp-silu |
| radial | 1 | large | 5 | 0.011 | 0.015 | -0.004 (-0.005, -0.004) | 5/5 | 1.000 | kan-rbf / mlp-relu |
| radial | 1 | small | 5 | 0.011 | 0.017 | -0.006 (-0.008, -0.005) | 5/5 | 1.000 | kan-rbf / mlp-relu |
| radial | 2 | large | 5 | 0.010 | 0.013 | -0.003 (-0.005, -0.001) | 5/5 | 1.000 | kan-rbf / mlp-relu |
| radial | 2 | small | 5 | 0.010 | 0.015 | -0.005 (-0.009, -0.000) | 4/5 | 1.000 | kan-rbf / mlp-relu |
| waves | 1 | large | 5 | 0.014 | 0.025 | -0.010 (-0.016, -0.005) | 5/5 | 1.000 | kan-rbf / mlp-relu |
| waves | 1 | small | 5 | 0.015 | 0.029 | -0.014 (-0.020, -0.009) | 5/5 | 1.000 | kan-spline / mlp-relu |
| waves | 2 | large | 5 | 0.016 | 0.020 | -0.005 (-0.007, -0.002) | 5/5 | 1.000 | kan-rbf / mlp-silu |
| waves | 2 | small | 5 | 0.015 | 0.019 | -0.004 (-0.008, 0.000) | 5/5 | 1.000 | kan-rbf / mlp-silu |
| yeast | 1 | large | 5 | 0.432 | 0.464 | -0.032 (-0.122, 0.057) | 3/5 | 1.000 | kan-rbf / mlp-relu |
| yeast | 1 | small | 5 | 0.446 | 0.476 | -0.030 (-0.087, 0.028) | 3/5 | 1.000 | kan-rbf / mlp-tanh |
| yeast | 2 | large | 5 | 0.462 | 0.493 | -0.030 (-0.071, 0.010) | 4/5 | 1.000 | kan-rbf / mlp-tanh |
| yeast | 2 | small | 5 | 0.559 | 0.486 | +0.073 (0.008, 0.138) | 0/5 | 1.000 | kan-rbf / mlp-tanh |

### Validation-selected KAN vs MLP, outside (extrapolation) test

| Task | Depth | Budget | n | KAN error | MLP error | KAN − MLP (95% CI) | KAN wins | Holm p |
|---|---:|---|---:|---:|---:|---|---:|---:|
| additive | 1 | large | 5 | 0.417 | 0.265 | +0.152 (0.070, 0.234) | 0/5 | 1.000 |
| additive | 1 | small | 5 | 0.549 | 0.215 | +0.334 (0.094, 0.574) | 0/5 | 1.000 |
| additive | 2 | large | 5 | 0.520 | 0.320 | +0.201 (0.030, 0.371) | 0/5 | 1.000 |
| additive | 2 | small | 5 | 0.505 | 0.291 | +0.214 (0.109, 0.319) | 0/5 | 1.000 |
| circadian_liver | 1 | large | 5 | 1.640 | 3.380 | -1.741 (-2.565, -0.916) | 5/5 | 1.000 |
| circadian_liver | 1 | small | 5 | 1.570 | 2.941 | -1.371 (-1.984, -0.759) | 5/5 | 1.000 |
| circadian_liver | 2 | large | 5 | 1.492 | 3.569 | -2.077 (-3.210, -0.944) | 5/5 | 1.000 |
| circadian_liver | 2 | small | 5 | 1.478 | 2.985 | -1.507 (-2.946, -0.069) | 5/5 | 1.000 |
| drug_combo | 1 | large | 12 | 0.776 | 0.829 | -0.053 (-0.148, 0.043) | 9/12 | 1.000 |
| drug_combo | 1 | small | 12 | 0.886 | 0.860 | +0.026 (-0.094, 0.146) | 5/12 | 1.000 |
| drug_combo | 2 | large | 12 | 0.932 | 0.814 | +0.118 (-0.029, 0.265) | 4/12 | 1.000 |
| drug_combo | 2 | small | 12 | 0.907 | 0.836 | +0.071 (-0.135, 0.276) | 3/12 | 1.000 |
| gb1_epistasis | 1 | large | 5 | 0.426 | 0.323 | +0.103 (0.033, 0.174) | 0/5 | 1.000 |
| gb1_epistasis | 1 | small | 5 | 0.411 | 0.320 | +0.090 (0.030, 0.151) | 0/5 | 1.000 |
| gb1_epistasis | 2 | large | 5 | 0.366 | 0.317 | +0.049 (0.006, 0.092) | 0/5 | 1.000 |
| gb1_epistasis | 2 | small | 5 | 0.366 | 0.314 | +0.052 (0.028, 0.075) | 0/5 | 1.000 |
| interaction | 1 | large | 5 | 0.550 | 0.089 | +0.460 (0.391, 0.530) | 0/5 | 1.000 |
| interaction | 1 | small | 5 | 0.628 | 0.108 | +0.520 (0.419, 0.621) | 0/5 | 1.000 |
| interaction | 2 | large | 5 | 0.666 | 0.101 | +0.565 (0.507, 0.624) | 0/5 | 1.000 |
| interaction | 2 | small | 5 | 0.652 | 0.085 | +0.567 (0.454, 0.681) | 0/5 | 1.000 |
| kink | 1 | large | 5 | 0.178 | 0.030 | +0.148 (0.092, 0.204) | 0/5 | 1.000 |
| kink | 1 | small | 5 | 0.181 | 0.028 | +0.153 (0.112, 0.194) | 0/5 | 1.000 |
| kink | 2 | large | 5 | 0.197 | 0.046 | +0.152 (0.110, 0.194) | 0/5 | 1.000 |
| kink | 2 | small | 5 | 0.240 | 0.033 | +0.206 (0.162, 0.250) | 0/5 | 1.000 |
| linear | 1 | large | 5 | 0.202 | 0.012 | +0.191 (0.175, 0.207) | 0/5 | 1.000 |
| linear | 1 | small | 5 | 0.217 | 0.013 | +0.204 (0.161, 0.246) | 0/5 | 1.000 |
| linear | 2 | large | 5 | 0.238 | 0.018 | +0.220 (0.187, 0.253) | 0/5 | 1.000 |
| linear | 2 | small | 5 | 0.229 | 0.028 | +0.201 (0.151, 0.251) | 0/5 | 1.000 |
| radial | 1 | large | 5 | 0.065 | 0.085 | -0.020 (-0.032, -0.008) | 5/5 | 1.000 |
| radial | 1 | small | 5 | 0.076 | 0.099 | -0.023 (-0.051, 0.004) | 4/5 | 1.000 |
| radial | 2 | large | 5 | 0.059 | 0.077 | -0.018 (-0.056, 0.019) | 3/5 | 1.000 |
| radial | 2 | small | 5 | 0.073 | 0.086 | -0.014 (-0.094, 0.067) | 3/5 | 1.000 |
| waves | 1 | large | 5 | 0.353 | 0.316 | +0.038 (-0.058, 0.133) | 2/5 | 1.000 |
| waves | 1 | small | 5 | 0.416 | 0.342 | +0.074 (-0.111, 0.259) | 1/5 | 1.000 |
| waves | 2 | large | 5 | 0.394 | 0.321 | +0.072 (-0.056, 0.201) | 1/5 | 1.000 |
| waves | 2 | small | 5 | 0.364 | 0.339 | +0.025 (-0.069, 0.120) | 1/5 | 1.000 |

### Six-model mean error by task (inside test; mean over seeds, depths and budgets)

| Task | mlp-tanh | mlp-relu | mlp-silu | kan-spline | kan-fourier | kan-rbf |
|---|---:|---:|---:|---:|---:|---:|
| linear | 0.007 | 0.008 | **0.005** | 0.010 | 0.020 | 0.010 |
| additive | 0.014 | 0.016 | **0.010** | 0.016 | 0.025 | 0.013 |
| interaction | 0.019 | 0.013 | **0.009** | 0.015 | 0.024 | 0.014 |
| waves | 0.034 | 0.026 | 0.029 | 0.017 | 0.026 | **0.014** |
| radial | 0.024 | 0.015 | 0.040 | 0.012 | 0.020 | **0.010** |
| kink | 0.018 | **0.009** | 0.035 | 0.014 | 0.022 | 0.014 |
| breast_cancer | 0.029 | 0.028 | **0.025** | 0.040 | 0.109 | 0.038 |
| circadian_blood | 1.012 | 1.013 | 1.019 | **1.011** | 1.017 | 1.012 |
| circadian_liver | 0.530 | **0.522** | 0.650 | 0.548 | 0.601 | 0.524 |
| diabetes | 0.705 | 0.702 | **0.695** | 0.720 | 0.786 | 0.734 |
| drug_combo | 0.608 | 0.597 | 0.617 | **0.578** | 0.647 | 0.583 |
| gb1_epistasis | 0.301 | **0.300** | 0.301 | 0.307 | 0.412 | 0.305 |
| mpsa_splicing | 0.598 | 0.606 | **0.597** | 0.608 | 0.638 | 0.598 |
| yeast | 0.482 | 0.480 | **0.477** | 0.484 | 0.589 | 0.478 |

Friedman test over 56 blocks: chi² = 89.1, p = 1.1e-17; Nemenyi critical difference = 1.01 ranks.

| Model | Mean rank |
|---|---:|
| kan-rbf | 2.75 |
| mlp-relu | 2.84 |
| mlp-silu | 3.00 |
| mlp-tanh | 3.34 |
| kan-spline | 3.50 |
| kan-fourier | 5.57 |
