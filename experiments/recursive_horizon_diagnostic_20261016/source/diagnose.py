"""Frozen replay over sealed observations/actions only. No simulator, search, training or label input."""
from pathlib import Path
import json,sys,hashlib,fcntl
import numpy as np
import torch
P=Path('/home/robot/research/projects/wm_symbolic')
TRAIN=P/'experiments/interaction_latent_repair_20261016'
AUDIT=P/'experiments/interaction_latent_repair_followup_audit_20261016'
OUT=P/'experiments/recursive_horizon_diagnostic_20261016'
sys.path.insert(0,str(TRAIN/'source'))
from runtime import load,normalize
from model import probs
from common import action
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def compact_labels(l):return {'y':l['y'],'v':l['v']}
def main():
 sources=set()
 def source(p):p=Path(p);sources.add(str(p));return read(p)
 selection=source(TRAIN/'models/canonical_matched_seed0/selected.json');sources.add(selection['path']);assert sha(selection['path'])==selection['sha256'];sources.add(str(TRAIN/'models/query_seed0.pt'));sources.add(str(TRAIN/'data/public/normalization.npz'))
 diagnosis=source(AUDIT/'results/reference_diagnosis.json');collections=source(TRAIN/'results/D2_collection.json');refs={r['root_id']:r for r in collections if r['condition']=='reference'};reference=[]
 for d in diagnosis:
  if d['arm']!='canonical_matched':continue
  ref=refs[d['root_id']];steps=[];last_after=None;observed=0
  for i,s in enumerate(d['reference_steps']):
   task=ref['tasks'][i];assert s['skill']==task['skill']and s['native_status']==task['status'];row={'horizon':i+1,'skill':s['skill'],'status':s['native_status'],'observed':'actual_probability'in s}
   if row['observed']:
    before=task['before'];after=task['after'];assert before and after
    if last_after is not None:assert last_after==before['public_path']
    last_after=after['public_path'];sources.update([before['public_path'],after['public_path']]);callfile=Path(before['public_path']).parent.parent/'_private/calls.json';calls=source(callfile);c=next(c for c in calls if c['role']=='task'and c['start_frame']==before['frame']and c['end_frame']==after['frame']);assert c['skill']==s['skill']and c['action']==action(s['skill'],None,96)
    assert s['actual_labels_evaluator_only']['y']==after['labels']['y']and s['actual_labels_evaluator_only']['v']==after['labels']['v'];assert all(after['labels']['v'])
    row.update(before_path=before['public_path'],after_path=after['public_path'],before_frame=before['frame'],after_frame=after['frame'],action=c['action'],recursive_probability=s['recursive_probability'],actual_readout_probability=s['actual_probability'],truth=compact_labels(after['labels']));observed+=1
   else:assert task.get('after')is None
   steps.append(row)
  reason=None if observed>=4 else'original_reference_has_three_skills'if len(steps)==3 else'first_skill_failed_later_three_unreached'
  reference.append({'source':'reference','root_id':d['root_id'],'planned_steps':len(steps),'observed_steps':observed,'four_step_eligible':observed>=4,'exclusion_reason':reason,'steps':steps,'probability_source':'preexisting sealed reference_diagnosis.json'})
 states=source(TRAIN/'data/D2/public/states.json');pathmap={r['path']:i for i,r in enumerate(states)};inputpath=TRAIN/'data/D2/public/inputs.npz';sources.add(str(inputpath));inputs=dict(np.load(inputpath))
 summaries=source(TRAIN/'results/closed_loop.json');episodes=[]
 for summ in summaries:
  if summ['arm']!='canonical_matched':continue
  root=summ['root_id'];base=TRAIN/'data/closed_loop'/root/'canonical_matched';calls=source(base/'_private/calls.json');decisions=source(base/'_private/decisions.json');assert len(calls)==summ['calls']and len(decisions)==len(calls)+1
  refpath=refs[root]['tasks'][0]['before']['public_path'];initial=decisions[0]['public_path'];sources.update([refpath,initial])
  with np.load(refpath)as a,np.load(initial)as b:assert set(a.files)==set(b.files)and all(np.array_equal(a[k],b[k])for k in a.files)
  rows=[]
  for i,c in enumerate(calls[:4]):
   before=decisions[i];after=decisions[i+1];assert c['role']=='task'and c['start_frame']==before['frame']and c['end_frame']==after['frame'];assert c['action']==action(c['skill'],None,96);assert all(after['actual_labels_evaluator_only']['v']);sources.update([before['public_path'],after['public_path']])
   rows.append({'horizon':i+1,'skill':c['skill'],'action':c['action'],'status':c['native']['status'],'observed':True,'before_path':before['public_path'],'after_path':after['public_path'],'before_frame':before['frame'],'after_frame':after['frame'],'truth':compact_labels(after['actual_labels_evaluator_only']),'actual_readout_probability':after['current_probability']})
  episodes.append({'source':'recorded_action_log','root_id':root,'planned_steps':'original recorded action stream; at most first4 evaluated','observed_steps':len(calls),'evaluated_steps':len(rows),'four_step_eligible':len(rows)>=4,'exclusion_reason':None if len(rows)>=4 else'original_collection_stopped_after_three_skills_no_fourth_observation','contains_refusal_in_first4':any(r['status']=='refused'for r in rows),'initial_reference_state_index':pathmap[refpath],'initial_readout_probability':decisions[0]['current_probability'],'steps':rows})
 lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);m,q,stats=load(0,'canonical_matched');initial_indices=[r['initial_reference_state_index']for r in episodes]
 with torch.inference_mode():
  z=m.encode(torch.tensor(inputs['objects'][initial_indices],device='cuda'),torch.tensor(inputs['scene'][initial_indices],device='cuda'));initial_read=probs(q(normalize(z,stats))).cpu().numpy();assert np.allclose(initial_read,np.array([r['initial_readout_probability']for r in episodes]),atol=1e-4,rtol=1e-4)
  for i,ep in enumerate(episodes):
   forward=z[i:i+1]
   for row in ep['steps']:
    forward=m.transition(forward,torch.tensor([row['action']],device='cuda'));row['recursive_probability']=probs(q(normalize(forward,stats)))[0].cpu().tolist()
 write(OUT/'results/trajectories.json',{'reference':reference,'recorded_action_logs':episodes})
 hashes={p:sha(p)for p in sorted(sources)};write(OUT/'configs/source_hashes.json',{'sha256':hashes});write(OUT/'results/inference_provenance.json',{'model':selection,'query_path':str(TRAIN/'models/query_seed0.pt'),'query_sha256':sha(TRAIN/'models/query_seed0.pt'),'reference_recursive_predictions_reused':True,'reference_recursion_source_code':str(AUDIT/'source/reference_diagnosis.py'),'new_frozen_inference_only_for_recorded_action_logs':True,'initial_state_encodings':len(episodes),'encoder_calls':1,'recursive_transitions':sum(len(r['steps'])for r in episodes),'real_successor_readout_reused_from_sealed_records':True,'initial_readout_max_abs_difference':float(np.max(np.abs(initial_read-np.array([r['initial_readout_probability']for r in episodes])))),'intermediate_observation_correction':False,'training_updates':0,'new_simulation_calls':0,'planning_search_calls':0,'threshold_or_model_selection_changes':False})
 print('FROZEN_DIAGNOSTIC_READY',len(reference),len(episodes),flush=True)
if __name__=='__main__':main()
