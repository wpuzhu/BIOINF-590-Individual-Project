| Task | Model | Inside, default | Inside, 24 h-aligned | Outside (forecast), default | Outside, 24 h-aligned |
|---|---|---:|---:|---:|---:|
| circadian_blood | mlp-tanh | 1.012 | 1.012 | — | — |
| circadian_blood | mlp-relu | 1.013 | 1.013 | — | — |
| circadian_blood | mlp-silu | 1.019 | 1.014 | — | — |
| circadian_blood | kan-spline | 1.011 | 1.012 | — | — |
| circadian_blood | kan-fourier | 1.017 | 1.016 | — | — |
| circadian_blood | kan-rbf | 1.012 | 1.011 | — | — |
| circadian_liver | mlp-tanh | 0.530 | 0.536 | 2.232 | 2.258 |
| circadian_liver | mlp-relu | 0.522 | 0.526 | 3.388 | 3.519 |
| circadian_liver | mlp-silu | 0.650 | 0.567 | 4.098 | 4.367 |
| circadian_liver | kan-spline | 0.548 | 0.546 | 1.453 | 1.435 |
| circadian_liver | kan-fourier | 0.601 | 0.692 | 1.442 | 1.406 |
| circadian_liver | kan-rbf | 0.524 | 0.543 | 1.661 | 1.566 |
| circadian_liver | *reference: mean / ridge / cosinor (inside)* | 1.142 | 1.125 | 0.487 | |
| circadian_liver | *reference: mean / ridge / cosinor (outside)* | 1.258 | 1.256 | 0.663 | |
| circadian_blood | *reference: mean / ridge / cosinor (inside)* | 1.092 | 1.061 | 0.999 | |
