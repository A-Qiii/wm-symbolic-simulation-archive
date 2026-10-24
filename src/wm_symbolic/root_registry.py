"""Resolve evaluator/reset roots through active episode provenance, not glob order."""
from pathlib import Path
import json

def active_root_directory(project,root_id,records=None):
 project=Path(project)
 if records is None:records=json.loads((project/'data/data_manifest.json').read_text())['records']
 matching=[r for r in records if r['root_id']==root_id]
 if not matching:raise KeyError('Root is not represented in the active manifest: '+root_id)
 # The first committed transition records the root's original collection namespace.
 version=Path(matching[0]['obs_before']).relative_to(project/'data/episodes').parts[0]
 root=project/'data/roots'/version/root_id
 if not (root/'identity.json').exists():
  # Later branches may be stored under main_v1 while their original reset root is retained.
  candidates=list((project/'data/roots').glob('*/'+root_id+'/identity.json'))
  if len(candidates)!=1:raise ValueError('Root provenance is ambiguous: '+root_id)
  root=candidates[0].parent
 identity=json.loads((root/'identity.json').read_text())
 if identity['root_id']!=root_id or identity['split']!=matching[0]['split'] or identity['environment']!=matching[0]['environment']:raise ValueError('Root provenance identity mismatch')
 return root
