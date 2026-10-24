"""Post-hoc single replacement; never changes selected model/actions or executes physics."""
from common import *
from runtime import *
from features import Encoder
from representation import transform
from model import probs
from metrics import discrete
import fcntl
def main():
 lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);enc=Encoder();norm=dict(np.load(E/'data/public/normalization.npz'));reports=[]
 for arm in ARMS:
  m,q,stats=load(0,arm)
  for summary in read(E/'results/closed_loop.json'):
   if summary['arm']!=arm:continue
   base=E/'data/closed_loop'/summary['root_id']/arm;decisions=read(base/'_private/decisions.json');calls=read(base/'_private/calls.json');j=3+COLORS.index(summary['target']);report={'root_id':summary['root_id'],'arm':arm,'physical_success':summary['physical_success'],'false_completion':summary['false_completion'],'replacements':[]};first=None
   for step,call in enumerate(calls):
    d=decisions[step];n=decisions[step+1];path=n['public_path'];fs=enc.encode([path]*8)[:1];f={k:np.stack([r[k]for r in fs])for k in fs[0]};o,s=transform(f,norm)
    with torch.no_grad():
     z=m.encode(torch.tensor(o,device='cuda'),torch.tensor(s,device='cuda'));p=probs(q(normalize(z,stats)))[0].cpu().numpy();delta=float(np.max(np.abs(p-n['current_probability'])));assert delta<.0001
     for choice in d['search']['plan'][1:]:z=m.transition(z,torch.tensor([action(choice['skill'],choice['target'],96)],device='cuda'))
     end=probs(q(normalize(z,stats)))[0];new=float(torch.minimum(end[j],end[6]))
    y=np.array(d['actual_labels_evaluator_only']['y']);v=np.array(d['actual_labels_evaluator_only']['v']);current_ok=bool(((discrete(np.array([d['current_probability']]))[0]==y)|~v).all());old=d['search']['score'];changed=old>=.8 and new<.8;row={'step':step,'skill':call['skill'],'native_status':call['native']['status'],'native_reason':call['native'].get('reason'),'current_all_valid_correct':current_ok,'original_planned_score':old,'replace_first_with_actual_observation_latent_score':new,'acceptance_removed':changed,'same_future_actions':d['search']['plan'][1:],'real_after_query_reencode_maxdiff':delta,'physical_reexecution':False};report['replacements'].append(row)
    if first is None and changed:first={'step':step,'current_readout_correct':current_ok,'kind':'consequential_first_transition_prediction_error','not_unique_component_isolation':True}
   if summary['false_completion']:first={'step':decisions[-1]['step'],'kind':'current_goal_readout_false_completion','physical_evaluator_goal':False}
   report['first_supported_error']=first;reports.append(report);print('DIAGNOSED',arm,summary['root_id'],first,flush=True)
 save(E/'results/single_substitution.json',reports)
if __name__=='__main__':main()
