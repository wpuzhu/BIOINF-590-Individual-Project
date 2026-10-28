"""Validate completed records without re-running training."""
import json,sys
from pathlib import Path
from collections import Counter
for arg in sys.argv[1:]:
 root=Path(arg);cfg=json.loads((root/'config.json').read_text());rows=json.loads((root/'results.json').read_text())
 expected=len(cfg['datasets'])*len(cfg['models'])*len(cfg['depths'])*len(cfg['budgets'])*len(cfg['seeds'])
 assert len(rows)==expected,(len(rows),expected)
 assert len({(r['dataset'],r['seed'],r['depth'],r['budget'],r['model']) for r in rows})==expected
 for r in rows:
  assert r['budget_relative_error']<=cfg['budget_tolerance']
  assert r['selected']['val_loss']==min(x['val_loss'] for x in r['candidates'])
  assert r['selected']['val_loss']==min(x['val_loss'] for x in r['selected']['history'])
  assert set(r['metrics'])==({'inside','outside'} if r['dataset'] in ['linear','additive','interaction','waves','radial','kink'] else {'inside'})
  assert abs(r['search_seconds']-sum(x['fit_seconds'] for x in r['candidates']))<1e-8
 print(root.name,len(rows),'records verified;',dict(Counter(r['selected']['stop_reason'] for r in rows)))
