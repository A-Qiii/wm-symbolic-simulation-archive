"""Two explicit channels: public model tensors and offline supervised targets."""
from pathlib import Path
from functools import lru_cache
import json
import numpy as np
import torch

def static_arguments(skill):
 pointer=next((i+1 for i,c in enumerate(['red','blue','pink']) if '_'+c+'_' in skill),0)
 source=next((i+1 for i,n in enumerate(['table','drawer','slider']) if skill.startswith('lift_') and skill.endswith('_'+n)),0)
 destination=1 if skill=='place_in_drawer' else (2 if skill=='place_in_slider' else 0)
 return [pointer,source,destination]

@lru_cache(maxsize=96)
def feature(path):
 with np.load(path) as z:return {k:z[k] for k in ['tokens','geometry','valid','robot','reconstruction']}

class Episodes:
 def __init__(self,project,split='train',limit_transitions=None,normalizer=None,event_count=3):
  p=Path(project);self.project=p;self.split=split;self.event_count=event_count;self.skills=[s['skill'] for s in json.loads((p/'configs/skills.json').read_text())]
  m=json.loads((p/'data/data_manifest.json').read_text());idx=json.loads((p/'data/features/dinov2_v1/index.json').read_text())
  self.cache={(x['episode_id'],x['step']):x['path'] for x in idx['items']};self.episodes=[];by_id={}
  records=[r for r in m['records'] if r['split']==split]
  if limit_transitions:records=records[:limit_transitions]
  for r in records:by_id.setdefault(r['episode_id'],[]).append(r)
  for eid,rows in by_id.items():
   rows.sort(key=lambda r:r['step']);assert [r['step'] for r in rows]==list(range(len(rows)))
   self.episodes.append({'episode_id':eid,'root_id':rows[0]['root_id'],'rows':rows,'labels':[json.loads(Path(r['labels_path']).read_text()) for r in rows]})
  if any(len(label['events'])<self.event_count for e in self.episodes for label in e['labels']):raise ValueError('Required process-event labels are missing')
  root_splits={}
  for r in m['records']:root_splits.setdefault(r['root_id'],set()).add(r['split'])
  if any(len(s)>1 for s in root_splits.values()):raise ValueError('Root leakage')
  self.root_ids=sorted(set(e['root_id'] for e in self.episodes));self.transitions=len(records)
  self.stats=normalizer
  self.matched=[i for i,e in enumerate(self.episodes) if any(r['collection_style']=='matched_order' for r in e['rows'])]
 def fit_stats(self):
  if self.split!='train':raise ValueError('Normalization fit forbidden outside training')
  sums=np.zeros(384,np.float64);squares=sums.copy();n=0;robots=[];counts=[np.zeros(c,np.int64) for c in [8,8,8,3,3]]
  for e in self.episodes:
   for step in range(len(e['rows'])+1):
    x=feature(self.cache[e['episode_id'],step]);tokens=x['tokens'][x['valid']].astype(np.float64);sums+=tokens.sum(0);squares+=(tokens**2).sum(0);n+=len(tokens);robots.append(x['robot'])
   labels=[e['labels'][0]['before']]+[r['after'] for r in e['labels']]
   for label in labels:
    for j,(v,ok) in enumerate(zip(label['categorical'],label['categorical_valid'])):
     if ok:counts[j][v]+=1
  mean=sums/n;std=np.sqrt(np.maximum(squares/n-mean**2,1e-6));robots=np.stack(robots)
  weights=[]
  for count in counts:
   active=count>0;w=np.ones(len(count),float);w[active]=count[active].sum()/active.sum()/count[active];weights.append(np.minimum(w,5).tolist())
  self.stats={'vision_mean':mean.tolist(),'vision_std':std.tolist(),'robot_mean':robots.mean(0).tolist(),'robot_std':np.maximum(robots.std(0),1e-3).tolist(),
   'class_counts':[x.tolist() for x in counts],'class_weights':weights,'fit_split':'train','fit_root_ids':self.root_ids,'fit_transitions':self.transitions}
  return self.stats
 def get(self,index):
  if self.stats is None:raise ValueError('Training normalization must be fitted explicitly')
  e=self.episodes[index];n=len(e['rows']);s=self.stats;vm=np.array(s['vision_mean'],np.float32);vs=np.array(s['vision_std'],np.float32);rm=np.array(s['robot_mean'],np.float32);rs=np.array(s['robot_std'],np.float32)
  fs=[feature(self.cache[e['episode_id'],t]) for t in range(n+1)];fs+=([fs[-1]]*(7-len(fs)))
  tokens=np.stack([(f['tokens'].astype(np.float32)-vm)/vs for f in fs]);robot=np.stack([(f['robot']-rm)/rs for f in fs])
  target=np.stack([np.concatenate([((f['reconstruction'].astype(np.float32).reshape(-1,384)-vm)/vs).reshape(-1),(f['robot']-rm)/rs]) for f in fs])
  labels=[e['labels'][0]['before']]+[x['after'] for x in e['labels']];labels+=([labels[-1]]*(7-len(labels)))
  rows=e['rows']+[e['rows'][-1]]*(6-n);results=e['labels']+[e['labels'][-1]]*(6-n)
  x={'tokens':tokens,'geometry':np.stack([f['geometry'] for f in fs]),'visible':np.stack([f['valid'] for f in fs]),'robot':robot,'target':target,
   'skill':np.array([self.skills.index(r['skill']) for r in rows],np.int64),'arguments':np.array([static_arguments(r['skill']) for r in rows],np.int64),
   'budget':np.array([r['executor']['budget_steps'] for r in rows],np.float32),'step_valid':np.arange(6)<n,
   'cat':np.array([x['categorical'] for x in labels],np.int64),'cat_valid':np.array([x['categorical_valid'] for x in labels],bool),
   'binary':np.array([x['binary'] for x in labels],np.float32),'binary_valid':np.array([x['binary_valid'] for x in labels],bool),
   'outcomes':np.array([[float(r['success'])]+r['events'][:self.event_count] for r in results],np.float32),'events_valid':np.array([r['events_valid'][:self.event_count] for r in results],bool)}
  return {k:torch.from_numpy(v) for k,v in x.items()}
 def batch(self,indices,device):
  xs=[self.get(int(i)) for i in indices];return {k:torch.stack([x[k] for x in xs]).to(device) for k in xs[0]}

def batch_indices(data,rng,batch_size=8):
 # 75% real distribution; 25% matched branches. Saved indices permit reuse.
 regular=rng.integers(len(data.episodes),size=batch_size-batch_size//4)
 special=rng.choice(data.matched or list(range(len(data.episodes))),size=batch_size//4)
 return np.concatenate([regular,special])
