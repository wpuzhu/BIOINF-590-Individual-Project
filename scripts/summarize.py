"""Aggregate actual run files; retain seeds, depth and budgets separately."""
import argparse,json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('directory');args=p.parse_args();root=Path(args.directory)
records=json.loads((root/'results.json').read_text());rows=[]
for r in records:
 for region,m in r['metrics'].items():
  rows.append({**{k:r[k] for k in ['dataset','seed','depth','budget','model','parameters','search_seconds']},'region':region,'fit_seconds':r['selected']['fit_seconds'],'epochs':r['selected']['epochs'],'stop_reason':r['selected']['stop_reason'],**m})
df=pd.DataFrame(rows);df.to_csv(root/'metrics.csv',index=False)
keys=['dataset','depth','budget','region','model']
metrics=[x for x in ['parameters','clean_rmse','rmse','macro_f1','balanced_accuracy','log_loss','fit_seconds','search_seconds','predict_us_per_sample','epochs'] if x in df]
summary=df.groupby(keys)[metrics].agg(['mean','std','count']);summary.to_csv(root/'summary.csv')
# Seed-paired differences versus MLP-SiLU. Bootstrap descriptive intervals with 3+ seeds only.
pairs=[]
import numpy as np
for group,g in df.groupby(['dataset','depth','budget','region']):
 for model in ['kan-spline','kan-fourier','kan-rbf']:
  a=g[g.model==model].set_index('seed');b=g[g.model=='mlp-silu'].set_index('seed');common=a.index.intersection(b.index)
  metric='clean_rmse' if 'clean_rmse' in a and a.clean_rmse.notna().any() else 'macro_f1'
  delta=(a.loc[common,metric]-b.loc[common,metric]).dropna().to_numpy()
  if not len(delta):continue
  lo=hi=None
  if len(delta)>=3:
   boot=np.random.default_rng(42).choice(delta,(10000,len(delta)),replace=True).mean(1)
   lo,hi=np.quantile(boot,[.025,.975])
  pairs.append(dict(zip(['dataset','depth','budget','region'],group))|dict(model=model,metric=metric,n=len(delta),mean_difference=delta.mean(),bootstrap_lo=lo,bootstrap_hi=hi))
pd.DataFrame(pairs).to_csv(root/'paired_differences.csv',index=False)
for (dataset,depth,budget,region),g in df.groupby(['dataset','depth','budget','region']):
 metric='clean_rmse' if 'clean_rmse' in g and g.clean_rmse.notna().any() else 'macro_f1'
 fig,ax=plt.subplots(figsize=(8,4));agg=g.groupby('model')[metric].agg(['mean','std'])
 ax.bar(agg.index,agg['mean'],yerr=agg['std'].fillna(0),color=['#2f7074' if m.startswith('kan') else '#ae653c' for m in agg.index],capsize=3)
 ax.set_ylabel(metric);ax.set_title(f'{dataset} | {depth} hidden layers | budget {budget} | {region}')
 ax.tick_params(axis='x',rotation=30);fig.tight_layout();fig.savefig(root/f'{dataset}_{depth}_{budget}_{region}.png',dpi=150);plt.close(fig)
# Validation objective against elapsed fitting time for one representative paired seed.
for dataset in df.dataset.unique():
 subset=[r for r in records if r['dataset']==dataset and r['seed']==min(df.seed) and r['depth']==min(df.depth)]
 fig,ax=plt.subplots(figsize=(8,4))
 for r in subset:
  h=r['selected']['history'];ax.plot([v['seconds'] for v in h],[v['val_loss'] for v in h],label=r['model'])
 ax.set(xlabel='Training wall time (seconds)',ylabel='Validation loss',title=dataset);ax.set_yscale('log');ax.legend();fig.tight_layout();fig.savefig(root/f'{dataset}_learning_curves.png',dpi=150);plt.close(fig)
print(f'Wrote metrics, grouped summaries, paired differences and figures for {len(records)} model runs to {root}')
