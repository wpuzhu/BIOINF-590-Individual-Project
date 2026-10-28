import json
from pathlib import Path
base=dict(models=['mlp-tanh','mlp-relu','mlp-silu','kan-spline','kan-fourier','kan-rbf'],terms=8,depths=[1,2],budgets=[1024],n_train=1000,noise=0.05,learning_rates=[0.003,0.01],weight_decay=0.0001,max_epochs=1500,check_every=10,patience=12,lr_patience=3,min_delta=0.001,budget_tolerance=0.05,threads=1,seeds=[11,22,33])
root=Path(__file__).resolve().parents[1]/'configs'
for name,datasets in [('simulation',['linear','additive','interaction','waves','radial','kink']),('biology',['yeast','breast_cancer'])]:
 cfg={**base,'datasets':datasets,'budgets':[4096] if name=='biology' else [1024]}
 (root/f'{name}.json').write_text(json.dumps(cfg,indent=2))
 pilot={**cfg,'max_epochs':300,'seeds':[11] if name=='simulation' else [11,22,33]}
 (root/f'{name}_pilot.json').write_text(json.dumps(pilot,indent=2))
 full={**cfg,'max_epochs':5000,'seeds':[11,22,33,44,55],'depths':[1,2,3],'budgets':[4096,8192] if name=='biology' else [1024,4096]}
 (root/f'{name}_full.json').write_text(json.dumps(full,indent=2))
