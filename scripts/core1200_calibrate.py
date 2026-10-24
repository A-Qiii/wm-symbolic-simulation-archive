"""Reserved C-root temperature diagnostics; never tune on D."""
from pathlib import Path
import json,hashlib,math,fcntl,os
import torch
from wm_symbolic.world_model import WorldModel
from wm_symbolic.core_dataset import CoreEpisodes
from wm_symbolic.core_detailed_evaluation import evaluate
P=Path('/home/robot/research/projects/wm_symbolic');R=P/'runs/core1200_v1';out=R/'evaluation/calibration'
def main():
 out.mkdir(exist_ok=False);lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 ckpath=R/'training_main/best.pt';ck=torch.load(ckpath,map_location='cpu',weights_only=False);assert ck['identity']['random_initialization'] and ck['identity']['skill_count']==6
 data=CoreEpisodes(P,'dev',normalizer=ck['normalization'],dev_role='calibration');assert len(data.root_ids)==8 and not set(data.root_ids)&set(ck['identity']['dev_roots'])
 model=WorldModel(skill_count=6,event_count=3).cuda().eval();model.load_state_dict(ck['model']);torch.manual_seed(81831);torch.set_num_threads(4)
 raw={k:[] for k in ['objects','drawer','slider','lights']};targets=[];valids=[];eraw=[];etarget=[];root_rows=[]
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
  for i in range(len(data.episodes)):
   b=data.batch([i],'cuda');enc=model.encode(b['tokens'],b['geometry'],b['visible'],b['robot']).repeat_interleave(8,0);act=model.action(b['skill'],b['arguments'],b['budget']).repeat_interleave(8,0);posts,_,priors,_=model.history(enc,act)
   for state,t in [(s,t) for t,s in enumerate(posts[:5])]+[(s,t+1) for t,s in enumerate(priors[:4])]:
    pred=model.symbols(state)
    for k in raw:raw[k].append(pred[k].float().cpu())
    targets.append(torch.cat([b['cat'][0,t],b['binary'][0,t].long()]).cpu());valids.append(torch.cat([b['cat_valid'][0,t],b['binary_valid'][0,t]]).cpu());root_rows.append(data.episodes[i]['root_id'])
   for t in range(4):eraw.append(model.outcome_logits(posts[t],act[:,t],priors[t])[:,1:].float().cpu());etarget.append(b['outcomes'][0,t,1:].cpu())
 raw={k:torch.stack(v) for k,v in raw.items()};target=torch.stack(targets);valid=torch.stack(valids);eraw=torch.stack(eraw);etarget=torch.stack(etarget)
 def objective(t):
  losses=[]
  for j in range(5):
   x=raw['objects'][:,:,j] if j<3 else raw['drawer' if j==3 else 'slider'];prob=(x/t).softmax(-1).mean(1);losses.append(-prob.clamp_min(1e-8).log().gather(1,target[:,j,None]).squeeze(1))
  binary=torch.nn.functional.binary_cross_entropy((raw['lights']/t).sigmoid().mean(1),target[:,5:].float(),reduction='none');values=torch.cat([torch.stack(losses,-1),binary],-1);return (values*valid).sum()/valid.sum()
 def fit(fn):
  logt=torch.zeros((),requires_grad=True);opt=torch.optim.LBFGS([logt],lr=.25,max_iter=80,line_search_fn='strong_wolfe')
  def closure():
   opt.zero_grad();loss=fn(logt.clamp(math.log(.05),math.log(100)).exp());loss.backward();return loss
  opt.step(closure);return float(logt.detach().clamp(math.log(.05),math.log(100)).exp())
 temp=fit(objective);event_temp=fit(lambda t:torch.nn.functional.binary_cross_entropy((eraw/t).sigmoid().mean(1),etarget))
 prob={k:((v/temp).sigmoid() if k=='lights' else (v/temp).softmax(-1)).mean(1) for k,v in raw.items()};curve=[]
 # Terminal-fact combinations only: three colors x {drawer,held} x {closed,open}.
 # This is explicitly not an LTL-process completion certification.
 scores=[];truths=[];eligible=[]
 for color in range(3):
  for loc in [3,6]:
   for drawer in [0,2]:
    scores.append(torch.minimum(prob['objects'][:,color,loc],prob['drawer'][:,drawer]));truths.append((target[:,color]==loc)&(target[:,3]==drawer));eligible.append(valid[:,color]&valid[:,3])
 scores=torch.stack(scores,-1);truths=torch.stack(truths,-1);eligible=torch.stack(eligible,-1)
 for tau in [.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,.99]:
  claims=(scores>=tau)&eligible;count=int(claims.sum());false=int((claims&~truths).sum());correct=int((claims&truths).sum());roots_with_claim={root_rows[i] for i in range(len(root_rows)) if claims[i].any()}
  curve.append({'threshold':tau,'claims':count,'false_claims':false,'false_claim_rate':false/count if count else None,'true_claims':correct,'true_fact_combinations':int((truths&eligible).sum()),'coverage':correct/max(1,int((truths&eligible).sum())),'claim_roots':len(roots_with_claim)})
 result={'checkpoint_sha256':hashlib.sha256(ckpath.read_bytes()).hexdigest(),'calibration':{'temperature':temp,'event_temperature':event_temp,'threshold':.8,'event_threshold':.9,'calibrated':False},'roots':data.root_ids,'transitions':data.transitions,'test_used':False,'network_weights_updated':False,'symbol_nll_before_temperature':float(objective(torch.tensor(1.))),'symbol_nll_after_temperature':float(objective(torch.tensor(temp))),'formal_completion_certified':False,'reason':'This core study reserves8 calibration roots; the existing >=10-root completion-support criterion is retained and cannot be met. Fixed-threshold closed-loop runs are diagnostic only, with independently scored physical outcomes.','thresholds_fixed_before_D':True,'terminal_fact_curve':curve,'curve_scope':'3 colors x2 ownership categories x2 drawer states on current and one-step prediction margins; not full LTL-process completion certification'}
 (out/'calibration.json').write_text(json.dumps(result,indent=2));torch.save({'raw':raw,'target':target,'valid':valid,'events':eraw,'event_target':etarget},out/'logits.pt')
 # Freeze selected checkpoint and source before the first D data collection.
 freeze={'checkpoint':str(ckpath),'checkpoint_sha256':result['checkpoint_sha256'],'selected_update':ck['update'],'selected_on':'12 C tuning roots only','calibration_roots':data.root_ids,'test_data_count_at_freeze':sum(r['split']=='test' for r in json.loads((P/'data/core1200_v1/data_manifest.json').read_text())['records']),'evaluation_sources':{name:hashlib.sha256((P/'src/wm_symbolic'/name).read_bytes()).hexdigest() for name in ['world_model.py','core_dataset.py','core_evaluation.py']},'planning_tasks_sha256':hashlib.sha256((R/'evaluation/tasks.json').read_bytes()).hexdigest(),'test_tuning_forbidden':True}
 freeze['evaluation_sources'].update({name:hashlib.sha256((P/'src/wm_symbolic'/name).read_bytes()).hexdigest() for name in ['core_detailed_evaluation.py','core_field_metrics.py']})
 # Extra summaries on the already selected C tuning cohort, no reselection.
 tuning=CoreEpisodes(P,'dev',normalizer=ck['normalization'],dev_role='tune')
 detail=evaluate(model,tuning);(out/'selected_C_tuning_detailed_metrics.json').write_text(json.dumps(detail,indent=2))
 assert freeze['test_data_count_at_freeze']==0
 (R/'evaluation/model_test_freeze.json').write_text(json.dumps(freeze,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
