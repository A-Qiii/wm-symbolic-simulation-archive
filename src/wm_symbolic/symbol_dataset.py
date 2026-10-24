from pathlib import Path
from functools import lru_cache
import json,numpy as np,torch
from wm_symbolic.dataset import Episodes,static_arguments

@lru_cache(maxsize=256)
def cached(path):
 with np.load(path) as z:return {k:z[k].copy() for k in ['mean_logits','sample_logits']}

class SymbolEpisodes:
 def __init__(self,project,split,perception_sha,normalization):
  p=Path(project);base=Episodes(p,split,normalizer=normalization,event_count=7)
  self.episodes=base.episodes;self.root_ids=base.root_ids;self.transitions=base.transitions;self.matched=base.matched;self.skills=base.skills
  index=json.loads((p/'data/features/symbol_beliefs'/perception_sha/split/'index.json').read_text())
  assert index['perception_checkpoint_sha256']==perception_sha
  self.cache={x['episode_id']:x['path'] for x in index['items']}
  assert set(self.cache)=={x['episode_id'] for x in self.episodes}
 def get(self,index):
  episode=self.episodes[index];rows=episode['rows'];labels=[episode['labels'][0]['before']]+[x['after'] for x in episode['labels']]
  assert len(rows)==6
  values={'symbol_logits':cached(self.cache[episode['episode_id']])['mean_logits'],'symbol_samples':cached(self.cache[episode['episode_id']])['sample_logits'],
          'skill':np.array([self.skills.index(r['skill']) for r in rows],np.int64),'arguments':np.array([static_arguments(r['skill']) for r in rows],np.int64),
          'budget':np.array([r['executor']['budget_steps'] for r in rows],np.float32),
          'cat':np.array([x['categorical'] for x in labels],np.int64),'cat_valid':np.array([x['categorical_valid'] for x in labels],bool),
          'binary':np.array([x['binary'] for x in labels],np.float32),'binary_valid':np.array([x['binary_valid'] for x in labels],bool),
          'outcomes':np.array([[float(x['success'])]+x['events'] for x in episode['labels']],np.float32),'events_valid':np.array([x['events_valid'] for x in episode['labels']],bool)}
  return {k:torch.from_numpy(v) for k,v in values.items()}
 def batch(self,indices,device):
  rows=[self.get(int(i)) for i in indices];return {k:torch.stack([r[k] for r in rows]).to(device) for k in rows[0]}
