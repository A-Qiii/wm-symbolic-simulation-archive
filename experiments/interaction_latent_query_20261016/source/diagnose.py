"""Read-only one-component substitutions on completed traces. Never tune models."""
from common import *
from features import Encoder
from representation import transform
from model import probs
from train_queries import load_query,normalize
from evaluate import discrete
import torch,fcntl
def main():
 lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);out={};recs=read(E/'results/D_records.json')
 for kind in['generic','ae','interaction']:
  for seed in[0,1]:
   rr=[r for r in recs if r['arm']==f'{kind}_seed{seed}'and r['status']=='success'];table={'both_correct':0,'true_successor_correct_predicted_wrong':0,'true_successor_wrong_predicted_correct':0,'both_wrong':0}
   for r in rr:
    y=np.array(r['true_after']['y']);v=np.array(r['true_after']['v']);a=bool(np.all((discrete(np.array([r['actual_successor_query']]))[0]==y)|~v));b=bool(np.all((discrete(np.array([r['predicted_successor_query']]))[0]==y)|~v));table['both_correct'if a and b else'true_successor_correct_predicted_wrong'if a else'true_successor_wrong_predicted_correct'if b else'both_wrong']+=1
   out[f'{kind}_seed{seed}']=table
 save(E/'results/readout_transition_decomposition.json',out)
 enc=Encoder();norm=dict(np.load(E/'data/public/normalization.npz'));runs=read(E/'results/closed_loop.json');diagnostics=[]
 for kind in['generic','ae','interaction']:
  m,q,stats=load_query(kind,0)
  for r in[r for r in runs if r['kind']==kind]:
   folder=E/'data/closed_loop'/r['root_id']/kind;dec=read(folder/'_private/decisions.json');actions=read(folder/'_private/actions.json')if(folder/'_private/actions.json').exists()else[];target=COLORS.index(r['target']);record={'kind':kind,'root_id':r['root_id'],'physical_success':r['physical_strict_goal_success'],'substitutions':[],'first_directly_supported_failure':None}
   if r['false_completion']:
    record['first_directly_supported_failure']={'step':dec[-1]['step'],'component':'current_semantic_readout','evidence':'Model declared current goal complete but physical goal false. Replacing only current goal query by evaluator truth removes the stop.','online_behavior_changed':False}
   for i,act in enumerate(actions):
    if i+1>=len(dec):continue
    before=dec[i];after=dec[i+1];path=after['public_path'];f=enc.encode([path]*8)[0];oo,ss=transform({k:v[None]for k,v in f.items()},norm)
    with torch.no_grad():
     z=m.encode(torch.tensor(oo,device='cuda'),torch.tensor(ss,device='cuda'));actualprob=probs(q(normalize(z,stats)))[0].cpu().numpy();zz=z
     for step in before['search']['plan'][1:]:zz=m.transition(zz,torch.tensor([action(step['skill'],step['target'],96)],device='cuda'))
     end=probs(q(normalize(zz,stats)))[0].cpu().numpy()
    same=float(np.max(np.abs(actualprob-np.array(after['current_probability']))));assert same<.0001;scores=[before['search']['score'],float(min(end[3+target],end[6]))];sub={'step':i,'skill':act['skill'],'native_status':act['native']['status'],'original_planned_goal_score':scores[0],'replace_only_first_transition_with_actual_successor_latent_goal_score':scores[1],'same_future_actions_used':before['search']['plan'][1:],'goal_acceptance_changes':scores[0]>=.8 and scores[1]<.8,'actual_successor_reencode_probability_max_difference':same,'new_physical_execution':False};record['substitutions'].append(sub)
    if sub['goal_acceptance_changes']and(record['first_directly_supported_failure']is None or i<record['first_directly_supported_failure']['step'])and not r['physical_strict_goal_success']:record['first_directly_supported_failure']={'step':i,'component':'first_executed_transition','evidence':'Holding the planned future actions and query head fixed, replacing only first predicted latent with actual observed successor removes the accepted goal. This localizes a consequential prediction error for this plan; it does not uniquely isolate all representation/readout errors.'}
   if not r['physical_strict_goal_success']and record['first_directly_supported_failure']is None:record['first_directly_supported_failure']={'component':'not_uniquely_localized','evidence':'Recorded trajectory and single-step substitution do not isolate a unique first cause; no model repair or extra test tuning performed.'}
   diagnostics.append(record)
 save(E/'results/closed_loop_single_substitution.json',diagnostics);print('DIAGNOSIS_COMPLETE',flush=True)
if __name__=='__main__':main()
