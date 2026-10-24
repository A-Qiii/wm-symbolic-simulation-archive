"""Additional inference on already sealed D2 records only; never select/tune/execute."""
from pathlib import Path
import sys
E=Path('/home/robot/research/projects/wm_symbolic/experiments/interaction_latent_repair_20261016');A=E.parent/'interaction_latent_repair_followup_audit_20261016'
sys.path.insert(0,str(E/'source'))
from common import *
from runtime import load,normalize,ARMS
from model import probs
from metrics import discrete
from plan import OPTIONS
import torch,fcntl
@torch.no_grad()
def beam_trace(m,q,stats,z,reference,target):
 acts=torch.tensor([action(s,j,96)for s,j in OPTIONS],device='cuda');paths=[()];states=z;out=[];ref=tuple(OPTIONS.index((s,COLORS.index(s.split('_')[1])if s.startswith('lift_')else None))for s in reference)
 for depth in range(1,5):
  nxt=m.transition(states.repeat_interleave(6,0),acts.repeat(len(states),1));pp=probs(q(normalize(nxt,stats)));values=torch.minimum(pp[:,3+target],pp[:,6]);newpaths=[path+(j,)for path in paths for j in range(6)];best=int(values.argmax());keep=values.argsort(descending=True)[:32];prefix=ref[:depth];pos=newpaths.index(prefix)if prefix in newpaths else None;out.append({'depth':depth,'best_score':float(values[best]),'reference_prefix_present':pos is not None,'reference_prefix_rank_if_present':int((values>values[pos]).sum())+1 if pos is not None else None,'reference_prefix_in_next_beam':pos in keep.tolist()if pos is not None else False,'stops_at_this_depth':bool(values[best]>=.8)})
  if values[best]>=.8:break
  states=nxt[keep];paths=[newpaths[i]for i in keep.tolist()]
 return out
def main():
 lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);states=read(E/'data/D2/public/states.json');paths={r['path']:i for i,r in enumerate(states)};inputs=dict(np.load(E/'data/D2/public/inputs.npz'));o=torch.tensor(inputs['objects'],device='cuda');s=torch.tensor(inputs['scene'],device='cuda');roots=read(E/'configs/D2_roots.json');refs={r['root_id']:r for r in read(E/'results/D2_collection.json')if r['condition']=='reference'};result=[]
 for arm in ARMS:
  m,q,stats=load(0,arm)
  with torch.no_grad():allz=m.encode(o,s);actual_probs=probs(q(normalize(allz,stats))).cpu().numpy()
  for root in roots:
   ref=refs[root['root_id']];initial=ref['tasks'][0]['before'];idx=paths[initial['public_path']];z=allz[idx:idx+1];j=COLORS.index(root['target']);dec=read(E/'data/closed_loop'/root['root_id']/arm/'_private/decisions.json')[0]
   with np.load(initial['public_path'])as x,np.load(dec['public_path'])as y:same=all(np.array_equal(x[k],y[k])for k in x.files)
   assert same,'Reference initial reset differs from matched closed loop; do not silently reuse';assert np.max(np.abs(actual_probs[idx]-dec['current_probability']))<.0001
   forward=z;steps=[]
   for t,skill in enumerate(root['normal_reference_skills']):
    act=torch.tensor([action(skill,None,96)],device='cuda')
    with torch.no_grad():forward=m.transition(forward,act);pp=probs(q(normalize(forward,stats)))[0].cpu().numpy()
    row={'step':t,'skill':skill,'recursive_probability':pp.tolist(),'recursive_goal_score':float(min(pp[3+j],pp[6]))};task=ref['tasks'][t]
    if task.get('after'):
     bi=paths[task['before']['public_path']];ai=paths[task['after']['public_path']]
     with torch.no_grad():one=m.transition(allz[bi:bi+1],act);onep=probs(q(normalize(one,stats)))[0].cpu().numpy();re=float((forward-allz[ai:ai+1]).square().mean());se=float((one-allz[ai:ai+1]).square().mean())
     row.update(native_status=task['status'],actual_probability=actual_probs[ai].tolist(),actual_goal_score=float(min(actual_probs[ai,3+j],actual_probs[ai,6])),single_step_from_actual_probability=onep.tolist(),single_step_goal_score=float(min(onep[3+j],onep[6])),recursive_latent_MSE=re,single_step_latent_MSE=se,actual_labels_evaluator_only=task['after']['labels'])
    else:row['native_status']=task['status']
    steps.append(row)
   last=ref['tasks'][-1];physical=False
   if last.get('after'):l=last['after']['labels'];physical=bool(l['y'][3+j]and l['y'][6]and l['v'][3+j]and l['v'][6])
   cp=dec['current_probability'];cv=np.array(initial['labels']['v']);cy=np.array(initial['labels']['y']);cur_ok=bool(((discrete(np.array([cp]))[0]==cy)|~cv).all())
   result.append({'arm':arm,'root_id':root['root_id'],'initial_open':root['drawer_reset_m']>.18,'current_goal_score':float(min(cp[3+j],cp[6])),'current_all_valid_correct':cur_ok,'chosen_search':dec['search'],'reference_sequence':root['normal_reference_skills'],'reference_physically_completed':physical,'reference_predicted_final_score':steps[-1]['recursive_goal_score'],'reference_steps':steps,'beam_trace':beam_trace(m,q,stats,z,root['normal_reference_skills'],j),'same_initial_RGB_and_proprio':same})
 save(A/'results/reference_diagnosis.json',result);save(A/'results/diagnostic_scope.json',{'training_updates':0,'new_physical_calls':0,'new_test_sets':0,'selected_models_modified':False,'threshold_modified':False,'source':'already sealed D2 real records','purpose':'post-hoc reference path and multi-step error diagnosis only'});print('REFERENCE_DIAG_COMPLETE',len(result),flush=True)
if __name__=='__main__':main()
