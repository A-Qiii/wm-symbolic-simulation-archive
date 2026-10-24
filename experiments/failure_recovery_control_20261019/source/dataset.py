from common import *
from wm_symbolic.dataset import Episodes
class NewEpisodes(Episodes):
 def __init__(self,split,arm=None):
  self.project=P;self.split=split;self.event_count=3;self.skills=SKILLS;self.stats=read(CELL/'configs/normalization.json');roots=[r for r in read(S/'data/roots_manifest.json')if r['split']==split];ids={r['root_id']for r in roots}
  rows=[r for r in read(S/'data/data_manifest.json')['records']if r['root_id']in ids];by=collections.defaultdict(list)
  for r in rows:by[r['episode_id']].append(r)
  self.cache={(f['episode_id'],f['step']):f['path']for f in read(S/'data/features/index.json')['items']if f['root_id']in ids};self.episodes=[]
  for root in roots:
   branches=[0,1]if arm=='nominal'else [0,2]if arm=='recovery'else[0,1,2]
   for b in branches:
    eid=f"{root['root_id']}_b{b}";rr=sorted(by[eid],key=lambda r:r['step']);assert len(rr)==6 and [r['step']for r in rr]==list(range(6));self.episodes.append({'episode_id':eid,'root_id':root['root_id'],'rows':rr,'labels':[read(r['labels_path'])for r in rr],'root':root,'branch':b})
  self.root_ids=[r['root_id']for r in roots];self.transitions=6*len(self.episodes);self.matched=list(range(len(self.episodes)))
