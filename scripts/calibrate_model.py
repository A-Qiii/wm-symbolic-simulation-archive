"""Fit temperatures and task confirmation thresholds on separate C roots only."""
from pathlib import Path
import argparse,json,hashlib,math
import numpy as np
import torch
from wm_symbolic.world_model import WorldModel
from wm_symbolic.dataset import Episodes
from wm_symbolic.gpu_phase import gpu_phase
from wm_symbolic.symbol_decoder import Calibration,decode,goal_score
from wm_symbolic.vision_cache import VisibleRegistry
from wm_symbolic.search import DFA
from wm_symbolic.symbol_transition import SymbolTransition
from score_closed_loop import facts_from
P=Path('/home/robot/research/projects/wm_symbolic');a=argparse.ArgumentParser();a.add_argument('--checkpoint',required=True);a.add_argument('--output',required=True);a.add_argument('--symbol-checkpoint');args=a.parse_args()
out=Path(args.output);out.mkdir(exist_ok=False);checkpoint=Path(args.checkpoint);ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
data=Episodes(P,'calibration',normalizer=ck['normalization'],event_count=7)
assert ck['identity']['mode'] in ['formal','one_step'] and ck['identity']['updates']==30000
assert data.transitions==360 and len(data.root_ids)==30
assert all(x.startswith('C_calibration_') for x in data.root_ids)
assert not set(data.root_ids)&set(ck['identity']['train_roots']+ck['identity']['dev_roots'])
torch.set_num_threads(4);torch.manual_seed(8139);model=WorldModel(event_count=7).eval();model.load_state_dict(ck['model']);samples=8
dynamics=model;symbol_sha=None
if args.symbol_checkpoint:
 symbol_path=Path(args.symbol_checkpoint);symbol_sha=hashlib.sha256(symbol_path.read_bytes()).hexdigest();symbol_ck=torch.load(symbol_path,map_location='cpu',weights_only=False)
 assert symbol_ck['identity']['perception_checkpoint_sha256']==hashlib.sha256(checkpoint.read_bytes()).hexdigest()
 assert symbol_ck['identity']['train_roots']==ck['identity']['train_roots']
 dynamics=SymbolTransition().eval();dynamics.load_state_dict(symbol_ck['model'])
compiled_tasks=json.loads((P/'configs/tasks/calibration/task_specs.json').read_text())
symbol_raw={k:[] for k in ['objects','drawer','slider','lights']};symbol_target=[];symbol_valid=[];event_raw=[];event_target=[];episodes=[]
def append_symbols(pred,target,valid):
 for key in symbol_raw:symbol_raw[key].append(pred[key].float().cpu())
 symbol_target.append(target.cpu());symbol_valid.append(valid.cpu())
with gpu_phase(P,'independent_C_calibration_logits'),torch.inference_mode():
 model.cuda()
 if dynamics is not model:dynamics.cuda()
 for i,episode in enumerate(data.episodes):
  b=data.batch([i],'cuda');reg=VisibleRegistry();visible=[]
  root_path=Path(episode['rows'][0]['obs_before']).parent.parent
  paths=[[root_path/f'root_observation_{j}.npz' for j in range(4)]]+[[Path(r['obs_after']).parent/f'boundary_history_{j}.npz' for j in range(4)] for r in episode['rows']]
  for boundary in paths:
   frames=[]
   for path in boundary:
    with np.load(path) as z:frames.append({k:z[k] for k in z.files})
   reg.update(frames);visible.append([j in reg.colour_ids for j in range(3)])
  with torch.autocast('cuda',dtype=torch.bfloat16):
   encoded=model.encode(b['tokens'],b['geometry'],b['visible'],b['robot']).repeat_interleave(samples,0)
   actions=model.action(b['skill'],b['arguments'],b['budget']).repeat_interleave(samples,0);posts,_,priors,_=model.history(encoded,actions)
   targets=torch.cat([b['cat'],b['binary'].long()],-1)[0];valids=torch.cat([b['cat_valid'],b['binary_valid']],-1)[0]
   current=[];events=[]
   for t in range(7):
    pred=model.symbols(posts[t]);append_symbols(pred,targets[t],valids[t]);current.append({k:v.float().cpu() for k,v in pred.items()})
   real_states=posts if dynamics is model else [SymbolTransition.from_symbols(model.symbols(s)) for s in posts]
   free=real_states[0]
   for t in range(6):
    prior=priors[t] if dynamics is model else dynamics.imagine(real_states[t],actions[:,t])[0]
    append_symbols(dynamics.symbols(prior),targets[t+1],valids[t+1]);free,_=dynamics.imagine(free,actions[:,t])
    if t+1 in [3,5]:append_symbols(dynamics.symbols(free),targets[t+1],valids[t+1])
    raw=dynamics.outcome_logits(real_states[t],actions[:,t],prior).float()[:,1:].cpu();event_raw.append(raw);event_target.append(b['outcomes'][0,t,1:].cpu());events.append(raw)
  episodes.append({'root_id':episode['root_id'],'episode_id':episode['episode_id'],'current':current,'events':events,'targets':targets.cpu(),'valids':valids.cpu(),'event_targets':b['outcomes'][0,:,1:].cpu(),'seen':visible})
  print('CALIBRATION_LOGITS',episode['episode_id'],flush=True)
 model.cpu();dynamics.cpu();torch.cuda.empty_cache()
raw={k:torch.stack(v) for k,v in symbol_raw.items()};target=torch.stack(symbol_target);valid=torch.stack(symbol_valid)
eraw=torch.stack(event_raw);etarget=torch.stack(event_target)
def temperature_fit(objective):
 log_t=torch.zeros((),requires_grad=True);optimizer=torch.optim.LBFGS([log_t],lr=.25,max_iter=80,line_search_fn='strong_wolfe')
 def closure():
  optimizer.zero_grad();loss=objective(log_t.clamp(math.log(.05),math.log(100.)).exp());loss.backward();return loss
 optimizer.step(closure);return float(log_t.detach().clamp(math.log(.05),math.log(100.)).exp())
def symbol_objective(t):
 losses=[]
 for j in range(5):
  logits=raw['objects'][:,:,j] if j<3 else raw['drawer' if j==3 else 'slider'];p=(logits/t).softmax(-1).mean(1)
  losses.append(-p.clamp_min(1e-9).log().gather(1,target[:,j,None]).squeeze(1))
 p=(raw['lights']/t).sigmoid().mean(1);binary=torch.nn.functional.binary_cross_entropy(p,target[:,5:].float(),reduction='none')
 values=torch.cat([torch.stack(losses,-1),binary],-1);return (values*valid).sum()/valid.sum()
temperature=temperature_fit(symbol_objective)
event_temperature=temperature_fit(lambda t:torch.nn.functional.binary_cross_entropy((eraw/t).sigmoid().mean(1),etarget))
for episode in episodes:
 episode['probs']=[{k:((v/temperature).sigmoid() if k=='lights' else (v/temperature).softmax(-1)).mean(0) for k,v in x.items()} for x in episode['current']]
 episode['event_probs']=[(x/event_temperature).sigmoid().mean(0) for x in episode['events']]
 episode['tasks']=[task for task in compiled_tasks if task['root_id']==episode['root_id']]
 assert len(episode['tasks'])==2
 episode['truth']=[]
 for t in range(7):
  lab={'categorical':episode['targets'][t,:5].tolist(),'categorical_valid':episode['valids'][t,:5].tolist(),'binary':episode['targets'][t,5:].tolist(),'binary_valid':episode['valids'][t,5:].tolist()}
  lights=episode['targets'][t,5:].tolist();events=episode['event_targets'][t-1].tolist() if t else [0,0,0,not lights[1],lights[1],not lights[0],lights[0]]
  episode['truth'].append(facts_from(lab,events))
thresholds=[.5,.55,.6,.65,.7,.75,.8,.85,.9,.93,.95,.97,.98,.99,.995,.999,1.0];event_thresholds=[.5,.6,.7,.8,.9,.95,.99]
curve=[]
for tau in thresholds:
 for event_tau in event_thresholds:
  cal=Calibration(temperature=temperature,event_temperature=event_temperature,threshold=tau,event_threshold=event_tau);claims=0;false=0;unverifiable=0;eligible=0;true_complete=0;true_claim=0;claim_roots=set();scorable_claims=0
  for episode in episodes:
   for task in episode['tasks']:
    dfa=DFA(task['dfa']);estimated=dfa.initial;actual=dfa.initial;actual_valid=True;estimate_valid=True
    for t in range(7):
     truth=episode['truth'][t]
     if any(truth.get(n) is None for n in dfa.props):actual_valid=False
     if actual_valid:actual=dfa.advance(actual,{n:truth[n] for n in dfa.props})
     root_events=torch.zeros(7) if t==0 else episode['event_probs'][t-1]
     facts,error=decode(episode['probs'][t],root_events,dfa.props,cal)
     unseen=lambda n:any(n.startswith(color+'_') and not episode['seen'][t][j] for j,color in enumerate(['red','blue','pink']))
     if error or any(unseen(n) for n in dfa.props):estimate_valid=False
     if estimate_valid:estimated=dfa.advance(estimated,facts)
     score=0. if any(unseen(n) for n in task['terminal_goals']) else float(goal_score(episode['probs'][t],task['terminal_goals'],cal))
     claim=estimate_valid and estimated in dfa.accepting and score>=tau
     if not actual_valid or any(truth.get(n) is None for n in task['terminal_goals']):
      if claim:claims+=1;unverifiable+=1
      continue
     eligible+=1;complete=actual in dfa.accepting and all(truth[n]==v for n,v in task['terminal_goals'].items());true_complete+=complete
     if claim:claims+=1;scorable_claims+=1;false+=not complete;true_claim+=complete;claim_roots.add(episode['root_id'])
  curve.append({'threshold':tau,'event_threshold':event_tau,'eligible_boundaries':eligible,'confirmed':claims,'false_confirmations':false,'false_confirmation_rate':false/claims if claims else None,
                'unverifiable_confirmations':unverifiable,'false_confirmation_rate_upper_bound':(false+unverifiable)/claims if claims else None,'scorable_confirmations':scorable_claims,
                'confirmed_roots':len(claim_roots),'true_complete_boundaries':true_complete,'true_confirmations':true_claim,'confirmation_coverage_of_true_completion':true_claim/true_complete if true_complete else None})
# Declared support requirement avoids declaring success from zero/few confirmations.
acceptable=[x for x in curve if x['scorable_confirmations']>=30 and x['confirmed_roots']>=10 and x['false_confirmation_rate_upper_bound']<=.05]
selected=max(acceptable,key=lambda x:(x['true_confirmations'],x['confirmed'],-x['threshold'],-x['event_threshold'])) if acceptable else None
cal=Calibration(temperature=temperature,event_temperature=event_temperature,threshold=selected['threshold'] if selected else 1.,event_threshold=selected['event_threshold'] if selected else 1.,calibrated=selected is not None)
result={'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'symbol_checkpoint_sha256':symbol_sha,'calibration':cal.as_dict(),'selected':selected,'curve':curve,
        'calibration_roots':data.root_ids,'root_count':30,'transitions':360,'test_data_used':False,'model_parameters_updated':False,
        'selection_rule':'Empirical false-confirmation upper bound (known false plus scoring-unverifiable)/all confirmations <=5%, at least30 scorable confirmations across10 C roots; maximize true confirmation coverage. Otherwise uncalibrated.',
        'support_rule_is_engineering_choice_not_statistical_guarantee':True,'status':'calibrated' if selected else 'insufficient_supported_confirmation_coverage'}
(out/'calibration.json').write_text(json.dumps(result,indent=2));torch.save({'episodes':episodes,'symbol_logits':raw,'symbol_targets':target,'symbol_valid':valid,'event_logits':eraw,'event_targets':etarget},out/'calibration_logits.pt')
print('CALIBRATION_COMPLETE',json.dumps({k:v for k,v in result.items() if k not in ['curve','calibration_roots']}))
