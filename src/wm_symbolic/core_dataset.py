"""Separate core manifest; inherit the audited public/target tensor contract."""
from pathlib import Path
import json
from wm_symbolic.dataset import Episodes

class CoreEpisodes(Episodes):
 def __init__(self,project,split='train',limit_transitions=None,normalizer=None,event_count=3,dev_role=None):
  p=Path(project);base=p/'data/core1200_v1';self.project=p;self.split=split;self.event_count=event_count
  self.skills=json.loads((p/'runs/core1200_v1/protocol.json').read_text())['skills']
  manifest=json.loads((base/'data_manifest.json').read_text());index=json.loads((base/'features/index.json').read_text())
  self.cache={(x['episode_id'],x['step']):x['path'] for x in index['items']};self.episodes=[];by={}
  rows=[r for r in manifest['records'] if r['split']==split and (dev_role is None or r['dev_role']==dev_role)]
  if limit_transitions:rows=rows[:limit_transitions]
  for row in rows:by.setdefault(row['episode_id'],[]).append(row)
  for eid,rr in by.items():
   rr.sort(key=lambda x:x['step']);assert [r['step'] for r in rr]==list(range(len(rr)))
   self.episodes.append({'episode_id':eid,'root_id':rr[0]['root_id'],'rows':rr,'labels':[json.loads(Path(r['labels_path']).read_text()) for r in rr]})
  root_splits={}
  for r in manifest['records']:root_splits.setdefault(r['root_id'],set()).add(r['split'])
  assert all(len(x)==1 for x in root_splits.values()),'Root leakage'
  self.root_ids=sorted({e['root_id'] for e in self.episodes});self.transitions=len(rows);self.stats=normalizer
  self.matched=list(range(len(self.episodes)))
