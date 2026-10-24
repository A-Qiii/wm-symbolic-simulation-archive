"""Offline assessment only; never imported by the acting model or search."""
from pathlib import Path
from collections import Counter,defaultdict
import json,gzip,hashlib
import numpy as np
import torch
from wm_symbolic.proposition_interface import decode,required_facts
from wm_symbolic.symbol_decoder import Calibration
from core1200_controller_audit import measure,summarize
P=Path('/home/robot/research/projects/wm_symbolic');R=P/'runs/proposition_interface_20261014'
def read(p):return json.loads(p.read_text())
def main():
 tasks=read(R/'tasks.json');cal=Calibration(**read(R/'evaluation/threshold_freeze.json')['calibration']);results=[];roots=[];failures=[];control=[];actual_readouts=[]
 for task in tasks:
  pair={};sample_probs={}
  for mode in ['legacy','proposition','reference']:
   d=R/'evaluation/closed_loop'/mode/task['task_id'];res=read(d/'result.json');actual=read(d/'score_private.json');runtime=read(d/'runtime.json');res['cohort']=task['cohort'];results.append(res)
   replay=read(d/'execution/evaluator_only/root_replay.json');assert replay['maximum_absolute_difference']<1e-5
   if mode!='reference':
    first=next(x['belief'] for x in runtime['trace'] if 'belief' in x);pair[mode]=torch.load(d/'planning_root_00_r0.pt',map_location='cpu',weights_only=False);sample_probs[mode]={k:torch.tensor(v) for k,v in first['symbol_probabilities'].items()}
    mapping={(0,0):actual['trace'][0]['facts']};reobs=0
    for x in actual['trace'][1:]:
     reobs+=x['request']['command']=='reobserve';mapping[(x['boundary'],reobs)]=x['facts']
    wrong=[];unknown=[]
    for x in runtime['trace']:
     if 'decoded_facts' not in x:continue
     truth=mapping[(x['boundary'],x['reobservations'])]
     for name,value in x['decoded_facts'].items():
      valid=truth.get(name) is not None
      if valid:
       actual_readouts.append({'task':task['task_id'],'cohort':task['cohort'],'method':mode,'boundary':x['boundary'],'reobservations':x['reobservations'],'name':name,'prediction':value,'truth':truth[name]})
       if value is None:unknown.append({'boundary':x['boundary'],'name':name})
       elif value!=truth[name]:wrong.append({'boundary':x['boundary'],'name':name,'predicted':value,'actual':truth[name]})
    if not res['physical_task_success']:
     plans=[]
     for f in sorted(d.glob('plan_*.json.gz')):
      pl=json.load(gzip.open(f,'rt'));plans.append({'file':str(f),'queries':pl['queries'],'rejections':pl['rejections'],'first_failed_edge':next((x for x in pl['edges'] if x.get('failed_details')),None)})
     failures.append({'task':task['task_id'],'cohort':task['cohort'],'method':mode,'status':res['status'],'first_current_wrong':wrong[0] if wrong else None,'current_unknowns':unknown,'search_evidence':plans,'first_physical_failure':next((x for x in actual['trace'] if x.get('result',{}).get('status')=='failed'),None),'no_unique_cause_claimed':True})
   for f in sorted((d/'execution/evaluator_only').glob('*/summary.json')):
    skill=f.parent.name.split('_',1)[1]
    if skill=='reobserve':continue
    x=read(f);a=measure(f.parent,skill,x['before'],x['after'],x['result']);a.update(task=task['task_id'],cohort=task['cohort'],method=mode);control.append(a)
  assert torch.equal(pair['legacy']['h'],pair['proposition']['h']) and torch.equal(pair['legacy']['z'],pair['proposition']['z']),'Initial latent differs between paired modes'
  assert torch.equal(pair['legacy']['cuda_rng'],pair['proposition']['cuda_rng'])
  maxdiff=max(float((sample_probs['legacy'][k]-sample_probs['proposition'][k]).abs().max()) for k in sample_probs['legacy']);assert maxdiff==0
  truth=read(R/'evaluation/closed_loop/legacy'/task['task_id']/'score_private.json')['trace'][0]['facts'];p={k:v.mean(0) for k,v in sample_probs['legacy'].items()};e=torch.zeros(3);old=decode(p,e,required_facts(task),cal,'legacy')[0];new=decode(p,e,required_facts(task),cal,'proposition')[0]
  roots.append({'task':task['task_id'],'root':task['root_id'],'cohort':task['cohort'],'latent_and_rng_exactly_equal':True,'probability_max_abs_difference':maxdiff,'scope':required_facts(task),'legacy':old,'proposition':new,'truth':{n:truth.get(n) for n in required_facts(task)},'rescued_correct':sum(old[n] is None and new[n] is not None and truth.get(n)==new[n] for n in old),'rescued_wrong':sum(old[n] is None and new[n] is not None and truth.get(n) is not None and truth[n]!=new[n] for n in old)})
 counts=[]
 for co in ['ABC_development','fresh_D']:
  for mode in ['legacy','proposition','reference']:
   rows=[x for x in results if x['cohort']==co and x['method']==mode];rr=[x for x in actual_readouts if x['cohort']==co and x['method']==mode];cr=[x for x in control if x['cohort']==co and x['method']==mode]
   counts.append({'cohort':co,'method':mode,'tasks':len(rows),'success':sum(x['physical_task_success'] for x in rows),'requests':sum(x['physical_skills'] for x in rows),'controls':sum(x['controls'] for x in cr),'claims':sum(x['model_claimed'] for x in rows),'false_claims':sum(x['false_confirmation'] for x in rows),'queries':sum(x['successor_queries'] for x in rows),'stops':dict(Counter(x['status'] for x in rows)),'readouts':{'valid':len(rr),'unknown':sum(x['prediction'] is None for x in rr),'wrong':sum(x['prediction'] is not None and x['prediction']!=x['truth'] for x in rr)},'controller':summarize(cr)})
 frozen=read(R/'evaluation/new_test_freeze.json');hashes={f:hashlib.sha256(Path(f).read_bytes()).hexdigest()==sha for f,sha in frozen['sha256'].items()};assert all(hashes.values())
 new_states=[]
 for task in tasks:
  if task['cohort']!='fresh_D':continue
  f=R/'roots'/task['root_id']/'branch0_initial_private.npz'
  if str(f) not in [x['path'] for x in new_states]:
   with np.load(f) as z:new_states.append({'path':str(f),'root':task['root_id'],'scene':z['scene_obs'].tolist()})
 # D native order is read from the existing private root's actual labels instead of assuming color order.
 from omegaconf import OmegaConf
 native=list(OmegaConf.load(P/'vendor/calvin/calvin_env/conf/scene/calvin_scene_D.yaml').objects.movable_objects);colors=['red','blue','pink']
 def positions(scene):
  lookup=dict(zip(native,np.asarray(scene)[6:].reshape(3,6)));return np.stack([lookup['block_'+c] for c in colors])
 historical=list((P/'data/roots').glob('*/D_*/reset_evaluator_only.npz'))+list((P/'data/core1200_v1/roots').glob('D_*/branch0_initial_private.npz'))
 for version in ['v1','v2','v3']:historical+=list((P/f'runs/physical_oracle_{version}/roots').glob('D_*/reset_evaluator_only.npz'))
 prior=[]
 for f in historical:
  with np.load(f) as z:prior.append((str(f),positions(z['scene_obs'])))
 near=[]
 for row in new_states:
  a=positions(row['scene'])
  for name,b in prior:
   dist=np.linalg.norm(a[:,:3]-b[:,:3],axis=1);yaw=np.abs((a[:,5]-b[:,5]+np.pi/2)%np.pi-np.pi/2)
   if np.all(dist<=.012) or np.all(dist<=.02) and np.all(yaw<=.25):near.append({'root':row['root'],'other':name})
  prior.append((row['root'],a))
 root_audit={'new_roots':len(new_states),'historical_roots':len(historical),'matches':near,'root_replay_all_equal_within1e-5':True}
 output={'complete':True,'counts':counts,'results':results,'paired_initial_states':roots,'actual_readouts':actual_readouts,'failure_evidence':failures,'controls':control,'frozen_hashes_unchanged':hashes,'actual_new_root_exclusion':root_audit,'scope':'Matched complete fact scope, shared selected thresholds, unchanged1000-step model and controller. No claim of unseenD environment. No subsequent repair or training.'}
 (R/'evaluation/RESULT.json').write_text(json.dumps(output,ensure_ascii=False,indent=2));print(json.dumps({'counts':counts,'root_audit':root_audit},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
