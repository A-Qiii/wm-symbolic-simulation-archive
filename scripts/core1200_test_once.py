from pathlib import Path
import json,hashlib,fcntl
import numpy as np
import torch
from wm_symbolic.world_model import WorldModel
from wm_symbolic.core_dataset import CoreEpisodes
from wm_symbolic.core_detailed_evaluation import evaluate
P=Path('/home/robot/research/projects/wm_symbolic');R=P/'runs/core1200_v1'
def main():
 freeze=json.loads((R/'evaluation/model_test_freeze.json').read_text());ckpath=Path(freeze['checkpoint']);assert hashlib.sha256(ckpath.read_bytes()).hexdigest()==freeze['checkpoint_sha256']
 for name,sha in freeze['evaluation_sources'].items():assert hashlib.sha256((P/'src/wm_symbolic'/name).read_bytes()).hexdigest()==sha
 out=R/'evaluation/test_once';out.mkdir(exist_ok=False);(out/'STARTED.json').write_text(json.dumps({'checkpoint_sha256':freeze['checkpoint_sha256'],'test_once':True}))
 lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4)
 ck=torch.load(ckpath,map_location='cpu',weights_only=False);data=CoreEpisodes(P,'test',normalizer=ck['normalization']);assert data.transitions==240 and len(data.root_ids)==20
 assert not set(data.root_ids)&set(ck['identity']['train_roots']+ck['identity']['dev_roots'])
 model=WorldModel(skill_count=6,event_count=3).cuda().eval();model.load_state_dict(ck['model']);result=evaluate(model,data)
 for prefix in ['current','prior','persistence','wrong_action']:
  fields=result['detailed']['field_metrics'][prefix];names=['red_location','blue_location','pink_location','drawer_state','slider_state']
  assert sum(fields[n]['correct'] for n in names)==result[prefix+'_correct']
  assert sum(fields[n]['valid'] for n in names)==result[prefix+'_facts']
 # Bootstrap whole roots (all three branches together), never independent rows.
 roots=list(result['root_metrics'].values());rng=np.random.default_rng(66001);intervals={}
 for key in ['current','prior','persistence','wrong_action','free1','free2','free3','free4','free_persist1','free_persist2','free_persist3','free_persist4']:
  for metric,num,den in [('accuracy','correct','facts'),('changed_accuracy','changed_correct','changed_facts')]:
   if key+'_'+den not in roots[0]:continue
   draws=[]
   for _ in range(2000):
    chosen=[roots[i] for i in rng.integers(len(roots),size=len(roots))];d=sum(r[key+'_'+den] for r in chosen)
    if d:draws.append(sum(r[key+'_'+num] for r in chosen)/d)
   intervals[key+'_'+metric]=np.quantile(draws,[.025,.975]).tolist()
 result.update(root_bootstrap_95_percentile_intervals=intervals,selected_update=ck['update'],checkpoint_sha256=freeze['checkpoint_sha256'],evaluation_complete=True,test_outcomes_used_for_model_changes=False)
 (out/'metrics.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['root_metrics','counterexamples']},indent=2))
if __name__=='__main__':main()
