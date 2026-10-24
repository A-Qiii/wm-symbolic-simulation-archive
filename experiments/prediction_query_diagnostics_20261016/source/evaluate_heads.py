"""Evaluate four preselected query heads once. C held-out roots are primary; D2 is a diagnostic backtest."""
from config import *
import sys,torch,csv
sys.path.insert(0,str(REP/'source'))
from model import Query,probs
from score_part1 import summarize

def metrics(pred,y,v,roots):
 pred=np.asarray(pred,bool);y=np.asarray(y,bool);v=np.asarray(v,bool);full=v.all(1);ok=((pred==y)|~v).all(1);groups={}
 for name,lo,hi in [('Holding',0,3),('In',3,6),('drawer',6,9)]:
  known=v[:,lo:hi].all(1);wrong=np.any(pred[:,lo:hi]!=y[:,lo:hi],1)&known
  groups[name]={'valid_states':int(known.sum()),'wrong_states':int(wrong.sum()),'false_positive_facts':int((pred[:,lo:hi]&~y[:,lo:hi]&v[:,lo:hi]).sum()),'false_negative_facts':int((~pred[:,lo:hi]&y[:,lo:hi]&v[:,lo:hi]).sum())}
 rootstats={r:{'N':int((roots==r).sum()),'valid_all_correct':int(ok[roots==r].sum()),'full9_N':int(full[roots==r].sum()),'full9_correct':int((ok&full)[roots==r].sum())}for r in sorted(set(roots))}
 macro=[x['full9_correct']/x['full9_N']for x in rootstats.values()if x['full9_N']]
 return {'N':len(y),'roots':len(rootstats),'full9_N':int(full.sum()),'full9_correct':int((full&ok).sum()),'full9_root_macro':float(np.mean(macro))if macro else None,'full9_roots':len(macro),'valid_facts':int(v.sum()),'correct_valid_facts':int(((pred==y)&v).sum()),'all_valid_correct':int(ok.sum()),'all_valid_root_macro':float(np.mean([x['valid_all_correct']/x['N']for x in rootstats.values()]))if rootstats else None,'groups':groups,'per_root':rootstats}

def main():
 torch.set_num_threads(4)
 # Verify the training freeze and checkpoint choices before opening validation labels.
 for p,h in read(E/'configs/training_frozen.json')['sha256'].items():assert sha(p)==h,p
 selected={};trainlogs={}
 for arm in ['original_coverage','intermediate_coverage']:
  for seed in [0,1]:
   name=f'{arm}_seed{seed}';s=read(E/'models'/name/'selected.json');t=read(E/'results'/f'head_training_{name}.json');assert t['updates']==1200 and t['sample_draws']==38400 and not t['world_model_loaded_or_updated']and not t['C_validation_or_D_read'];best=min(t['log'],key=lambda x:x['C_selection_root_macro_loss']);assert s['path'].endswith(f'update{best["update"]:04d}.pt');assert sha(s['path'])==s['sha256'];selected[name]=s;trainlogs[name]=t
 for seed in[0,1]:assert trainlogs[f'original_coverage_seed{seed}']['initial_query_hash']==trainlogs[f'intermediate_coverage_seed{seed}']['initial_query_hash']
 save(E/'configs/heads_frozen_before_validation.json',{'frozen_utc':now(),'selection_complete_before_C_validation_read':True,'selected_heads':selected,'sha256':{s['path']:s['sha256']for s in selected.values()}})
 baseline=torch.load(REP/'models/query_seed0.pt',map_location='cpu',weights_only=True);stats=baseline['normalization'];heads={'original_fixed_head':baseline['state']}
 for name,s in selected.items():heads[name]=torch.load(s['path'],map_location='cpu',weights_only=True)['state']
 cv=dict(np.load(E/'data/evaluator/C_validation.npz'));ids=cv['state_ids'];roots=cv['root_ids'];y=cv['y'].astype(bool);v=cv['v'].astype(bool)
 pairs=read(REP/'data/public/pairs.json');outcomes=read(REP/'data/_private/outcomes.json');postplace={p['after']for p in pairs if p['split']=='dev'and p['skill']=='place_in_drawer'and outcomes[p['id']]['status']=='success'}
 strata={'holding_open':v[:,:3].all(1)&y[:,:3].any(1)&v[:,7]&y[:,7],'holding_closed':v[:,:3].all(1)&y[:,:3].any(1)&v[:,6]&y[:,6],'post_place':np.array([int(i)in postplace for i in ids])};strata['other_states']=~np.stack(list(strata.values())).any(0)
 raw=dict(np.load(E/'data/public/part1_latents.npz'));part1=read(E/'results/part1_scored_records.json');truth={r['trajectory_id']:r for r in read(E/'data/evaluator/part1_truth.json')};allresults={};stateoutput=[];pairoutput=[];pred_by_head={};d2records={}
 with torch.inference_mode():
  for name,state in heads.items():
   q=Query();q.load_state_dict(state);q.eval();p=probs(q(torch.tensor(cv['z']))).numpy();pred=discrete(p);pred_by_head[name]=pred
   c={'overall':metrics(pred,y,v,roots),'strata':{k:metrics(pred[mask],y[mask],v[mask],roots[mask])for k,mask in strata.items()}}
   for j,i in enumerate(ids):stateoutput.append({'head':name,'state_id':int(i),'root_id':roots[j],'strata':[k for k,mask in strata.items()if mask[j]],'truth':y[j],'valid':v[j],'probability':p[j],'prediction':pred[j]})
   probabilities={method:probs(q((torch.tensor(raw[method])-stats['mean'])/stats['std'])).numpy()for method in ['actual','one_step','recursive']};records=[]
   for j,old in enumerate(part1):
    r={k:old[k]for k in ['trajectory_id','source','root_id','step','skill','truth','valid','status','one_step_latent_MSE','recursive_latent_MSE']}
    for method in probabilities:r[method+'_probability']=probabilities[method][j].tolist();r[method+'_labels']=discrete(probabilities[method][j]).tolist();r[method+'_ok']=r[method+'_labels']==r['truth']
    records.append(r)
   d2records[name]=records
   normal=[r for r in records if r['source']=='reference'and truth[r['trajectory_id']]['four_step_eligible']];refusal=[r for r in records if r['source']=='recorded_action_logs'and truth[r['trajectory_id']]['four_step_eligible']and truth[r['trajectory_id']]['contains_refusal']]
   allresults[name]={'C_validation':c,'D2_diagnostic':{'normal':summarize(normal),'refusal':summarize(refusal)}}
   if name=='original_fixed_head':
    err=max(float(np.abs(probabilities[method][j]-np.array(old[method+'_probability'])).max())for j,old in enumerate(part1)for method in probabilities);assert err<1e-4,err
 # Paired C counts describe changes on the same state, separated by validity and stratum.
 for seed in[0,1]:
  a=pred_by_head[f'original_coverage_seed{seed}'];b=pred_by_head[f'intermediate_coverage_seed{seed}'];ao=((a==y)|~v).all(1);bo=((b==y)|~v).all(1)
  for stratum,mask in [('all',np.ones(len(y),bool))]+list(strata.items()):
   for label,eligible in [('full9',v.all(1)),('all_valid',np.ones(len(y),bool))]:
    m=mask&eligible;pairoutput.append({'seed':seed,'stratum':stratum,'label_scope':label,'N':int(m.sum()),'control_wrong_coverage_right':int((m&~ao&bo).sum()),'control_right_coverage_wrong':int((m&ao&~bo).sum()),'both_right':int((m&ao&bo).sum()),'both_wrong':int((m&~ao&~bo).sum())})
 save(E/'results/head_evaluation.json',{'evaluated_utc':now(),'C_validation_primary':True,'D2_diagnostic_repeatedly_inspected':True,'new_D_evaluation':False,'new_physics':False,'thresholds_changed':False,'query_capacity_changed':False,'baseline_probability_reproduction_error':err,'heads':allresults,'C_paired':pairoutput})
 save(E/'results/C_validation_state_predictions.json',stateoutput);save(E/'results/D2_diagnostic_head_predictions.json',d2records)
 with(R/'C_validation_paired_states.csv').open('w')as f:
  w=csv.DictWriter(f,fieldnames=list(pairoutput[0]));w.writeheader();w.writerows(pairoutput)
 for name,x in allresults.items():print(name,json.dumps(x['C_validation'],ensure_ascii=False),flush=True)
if __name__=='__main__':main()
