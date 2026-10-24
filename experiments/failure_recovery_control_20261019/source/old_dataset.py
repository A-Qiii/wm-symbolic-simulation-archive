from common import *
SIZE=40
from wm_symbolic.dataset import Episodes

class CoreEpisodes(Episodes):
 def __init__(self,project,split='train',normalizer=None,event_count=3,roots_override=None):
  assert Path(project)==P
  self.project=P;self.split=split;self.event_count=event_count
  members=read(B/'configs/fixed_design.json')['members']
  name={'train':'train'+str(SIZE),'common20':'train20','validation':'validation10','test':'test10','externalC':'C20'}[split]
  ids=set(members[name] if roots_override is None else roots_override)
  assert ids<=set(members[name])
  self.skills=read(P/'runs/core1200_v1/protocol.json')['skills']
  rows=[r for r in read(P/'data/core1200_v1/data_manifest.json')['records'] if r['root_id'] in ids]
  self.cache={(x['episode_id'],x['step']):x['path']for x in read(P/'data/core1200_v1/features/index.json')['items']if x['root_id']in ids}
  by={};self.episodes=[]
  for r in rows:by.setdefault(r['episode_id'],[]).append(r)
  for eid,rr in sorted(by.items()):
   rr.sort(key=lambda r:r['step']);assert [r['step']for r in rr]==[0,1,2,3]
   self.episodes.append({'episode_id':eid,'root_id':rr[0]['root_id'],'rows':rr,'labels':[read(r['labels_path'])for r in rr]})
  self.root_ids=sorted(ids);self.transitions=len(rows);self.stats=normalizer;self.matched=list(range(len(self.episodes)))
  assert len(self.episodes)==3*len(ids)and self.transitions==12*len(ids)
  assert set(e['root_id']for e in self.episodes)==ids
