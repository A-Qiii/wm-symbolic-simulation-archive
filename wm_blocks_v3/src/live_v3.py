"""Same new initial layouts/candidates: frozen v1 vs v3, seed0 actual tasks."""
from paths import GPU_LOCK
from paths import ROOT as B, RUNTIME as P
import json,hashlib,fcntl
from pathlib import Path
import numpy as np,torch
from models import RGBState,Dynamics
from models_v3 import DiagonalStateDynamics
from rgb_front import RGBFront
from live import Worker,Runtime,choose,programs,same_goal
from evaluate_v3 import models
def read(p):return json.load(open(p))
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
class RT(Runtime):
 def __init__(self,rec):
  self.rec=rec;self.norm=np.load(rec['normalization']);self.est=RGBState(1567).cuda();self.est.load_state_dict(torch.load(rec['state_checkpoint'],map_location='cuda',weights_only=False)['model']);self.est.eval().requires_grad_(False);self.net=(Dynamics('motion_matrix')if rec['version']=='v1'else DiagonalStateDynamics()).cuda();self.net.load_state_dict(torch.load(rec['checkpoint'],map_location='cuda',weights_only=False)['model']);self.net.eval().requires_grad_(False)
def specs(manifest):
 for g in manifest['groups']:
  es=[e for e in manifest['episodes']if e['group']==g['id']];find=lambda name:next(e for e in es if e['recipe']==name)
  for loaded in [True,False]:
   e=find('B_loaded_reverse'if loaded else'B_empty_reverse');yield {'id':e['id'],'group':g['id'],'layout':e['layout'],'goal':{'large':0,'small':0 if loaded else 1,'always_small':None if loaded else 1},'task':'transition','stratum_private':'loaded'if loaded else'empty','context':[],'purpose':'withheld reverse transfer','split':'test'}
  for stacked in [True,False]:
   e=find('A_loaded_forward'if stacked else'A_empty_forward')
   for goal in [1,2]:yield {'id':g['id']+f'_basic_{"stacked"if stacked else"beside"}_goal{goal}','group':g['id'],'layout':e['layout'],'goal':{'large':1,'small':0 if goal==1 else 1,'always_small':0 if goal==1 else None},'task':'basic','goal_id':goal,'stratum_private':'loaded'if stacked else'empty','context':[],'purpose':'same candidate task generalization','split':'test'}
def main():
 assert(B/'reports/v3/confirmation/complete.json').exists();manifest=read(B/'configs/v3_confirmation_manifest.json');out=B/'reports/v3/live';out.mkdir(parents=True,exist_ok=True)
 if(out/'complete.json').exists():return
 for n in ['block_env.py','kinematic_guard.py','physics_monitor.py']:assert(B/'src'/n).read_bytes()==(B/'data/v1/frozen_source'/n).read_bytes()
 front=RGBFront();worker=Worker(out);records=[]
 try:
  for rec in [r for r in models()if r['seed']==0 and r['version']in ['v1','v3']]:
   rt=RT(rec)
   for spec in specs(manifest):
    folder=out/rec['version']/spec['id']
    if(folder/'policy.json').exists()and(folder/'score_private.json').exists():records.append({'version':rec['version'],'case':spec['id'],'score_file':str(folder/'score_private.json')});continue
    if folder.exists():raise RuntimeError('Incomplete actual task retained: '+str(folder))
    reply=worker.call({'command':'start','folder':str(folder),'spec':spec,'capture':False});prefix=[];log=[];stop='invalid_initialization'
    if reply['initial_valid']:
     while True:
      raw,_=front.extract(reply['public_paths']);state=rt.observe(raw);current=rt.symbols(state);selected,candidates=choose(rt,state,prefix,programs(spec['task']),spec['goal']);entry={'prefix':prefix.copy(),'control_count':reply['control_count'],'current':current,'selected':selected,'candidates':candidates,'public_paths':reply['public_paths']}
      references=[e for e in manifest['episodes']if e['layout']==spec['layout']and e['actions'][:len(prefix)]==prefix]
      if references:
       p=B/'cache/v3/confirmation'/(references[0]['id']+f'__b{len(prefix)}.npz')
       if p.exists():entry['feature_max_abs_diff_vs_reference']=float(np.max(np.abs(raw-np.load(p)['features'])))
      log.append(entry)
      if same_goal(current,spec['goal']):stop='goal_reached';break
      if spec['goal']['always_small']is not None and current['region'][1]!=spec['goal']['always_small']:stop='model_current_constraint_violation';break
      if selected is None:stop='no_feasible_prediction';break
      a=selected['program'][len(prefix)];reply=worker.call({'command':'execute','action':a});prefix.append(a)
      if reply['status']!='completed':stop='low_level_skill_failure';break
      assert len(prefix)<=4
    save(folder/'policy.json',{'model':rec,'task_spec':spec,'selected_actions':prefix,'boundaries':log,'stop':stop,'no_GT_model_input':True});worker.call({'command':'finish','stop':stop});records.append({'version':rec['version'],'case':spec['id'],'score_file':str(folder/'score_private.json')});save(B/'STATUS.json',{'version':'v3','stage':'actual_confirmation_tasks','completed':len(records),'planned':240,'model_version':rec['version'],'case':spec['id'],'stop':stop});print('V3_LIVE',len(records),rec['version'],spec['id'],stop,flush=True)
   del rt;torch.cuda.empty_cache()
 finally:worker.close();front.close()
 scores=[{**r,**read(r['score_file'])}for r in records];save(out/'all_results.json',scores);save(out/'complete.json',{'planned':240,'completed':len(scores),'success':sum(r['success']for r in scores),'versions':['v1','v3'],'seed':0,'new_test_layouts':20,'no_test_tuning':True})
if __name__=='__main__':
 lock=GPU_LOCK.open('r');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
 try:main()
 finally:fcntl.flock(lock,fcntl.LOCK_UN)
