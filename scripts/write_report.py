"""Build the Markdown results report from completed run records."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def ms(v):
 a=np.asarray(v,dtype=float);return f'{a.mean():.4f} ± {a.std(ddof=1):.4f}' if len(a)>1 else f'{a[0]:.4f}'
text=['# Measured results','',
'These are actual CPU runs of the code in this repository. Main runs use three seeds, two hidden depths, two candidate learning rates and a maximum of 1,500 epochs per candidate. Pilot runs have a 300-epoch limit and are retained separately. The larger full configs have not been run. The runner trains models sequentially, but the execution host is shared, so timing results are exploratory. Pilot timings were collected while the two pilot studies ran concurrently and should not be used for clean runtime comparisons.','',
'## Biological held-out validation','',
'Each cell uses a stratified 60/20/20 split, training-only standardization and validation-only learning-rate/checkpoint selection. Scores below are internal held-out macro-F1, mean ± SD across three split seeds. They do not establish external cohort or homology-aware generalization.','',
'| Dataset | Hidden layers | Model | Parameters | Macro-F1 | Fit seconds | Search seconds | Plateau / runs |',
'|---|---:|---|---:|---:|---:|---:|---:|']
rows=json.loads((ROOT/'results/biology/results.json').read_text())
for dataset in ['yeast','breast_cancer']:
 for depth in [1,2]:
  for model in ['mlp-tanh','mlp-relu','mlp-silu','kan-spline','kan-fourier','kan-rbf']:
   a=[r for r in rows if r['dataset']==dataset and r['depth']==depth and r['model']==model]
   text.append(f"| {dataset} | {depth} | {model} | {a[0]['parameters']} | {ms([r['metrics']['inside']['macro_f1'] for r in a])} | {ms([r['selected']['fit_seconds'] for r in a])} | {ms([r['search_seconds'] for r in a])} | {sum(r['selected']['stop_reason']=='validation_plateau' for r in a)} / {len(a)} |")
text += ['','Fit seconds describe the selected candidate. Search seconds include both learning-rate candidates. Timing includes validation and checkpoint copies; see algorithms.md. Hardware is a shared execution environment, so these timings are descriptive and should be repeated on a dedicated machine before efficiency claims. All biological main-run selected fits reached the validation-plateau rule.','',
'## Synthetic function recovery','',
'Noise-free test RMSE is reported separately for inside [-1,1]^2 and outside [-1.5,1.5]^2 excluding the training square. Lower is better. Means and SDs are across three seeds; training noise SD is 0.05. Same formulas as the HTML, but a separate Python engine.','']
path=ROOT/'results/simulation/results.json'
if path.exists():
 sim=json.loads(path.read_text())
 for region in ['inside','outside']:
  text += [f'### {region.capitalize()}-range RMSE','','| Target | Hidden layers | MLP tanh | MLP ReLU | MLP SiLU | KAN spline | KAN Fourier | KAN RBF |','|---|---:|---:|---:|---:|---:|---:|---:|']
  for target in ['linear','additive','interaction','waves','radial','kink']:
   for depth in [1,2]:
    cells=[]
    for model in ['mlp-tanh','mlp-relu','mlp-silu','kan-spline','kan-fourier','kan-rbf']:
     a=[r for r in sim if r['dataset']==target and r['depth']==depth and r['model']==model]
     cells.append(ms([r['metrics'][region]['clean_rmse'] for r in a]))
    text.append('| '+target+' | '+str(depth)+' | '+' | '.join(cells)+' |')
  text.append('')
 reasons=Counter(r['selected']['stop_reason'] for r in sim)
 text += ['### Optimization status','',str(dict(reasons)),'',
 'Epoch-limit runs are **not converged results**. Inspect per-run histories and raise limits before interpreting an apparent representational disadvantage. No test data were used to decide which checkpoint or learning rate to select.','']
text += ['## Selected figures','','![Yeast validation-time curves](../results/biology/yeast_learning_curves.png)','','![Breast cancer validation-time curves](../results/biology/breast_cancer_learning_curves.png)','','![Periodic target validation-time curves](../results/simulation/waves_learning_curves.png)','','![Localized target validation-time curves](../results/simulation/radial_learning_curves.png)','','## What these experiments can support','',
'- Differences are conditional on a fixed-grid KAN implementation, chosen parameter budgets and a two-value learning-rate search. No result establishes that all KANs or MLPs behave the same way.',
'- A local basis can help recover a localized toy response, but extrapolation and real biological prediction are separate questions.',
'- Compare depth within budget as well as model family within depth. Hidden-layer count does not equalize the number or flexibility of nonlinear operations.',
'- Three resampled splits are too few for strong statistical claims. Paired bootstrap intervals in CSV files are descriptive and do not account for overlapping test subjects across split seeds.',
'- Real periodic gene expression, localized dose windows, biological extrapolation and independent external cohorts remain proposed extensions.','',
'## Inspect and reproduce','',
'Each results directory includes metrics.csv, summary.csv, paired_differences.csv, validation-time curves, per-model histories, split metadata and environment.json. Training checkpoints are generated locally and excluded from Git. Re-run the configs to regenerate them. Initial environment records may have null commit IDs because collection preceded repository initialization.','']
(ROOT/'docs/results.md').write_text('\n'.join(text))
