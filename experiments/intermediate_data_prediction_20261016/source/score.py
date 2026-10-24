"""Evaluator-only scoring after predictions and checkpoint choices are sealed."""
from config import *
import argparse,csv,sys,torch
sys.path.insert(0,str(REP/'source'))
from model import Query,probs
def metric(p,y,v,roots):
 p=discrete(p);y=np.array(y,bool);v=np.array(v,bool);roots=np.array(roots);full=v.all(1);ok=((p==y)|~v).all(1);rs={r:{'N':int((roots==r).sum()),'full9_N':int((full&(roots==r)).sum()),'full9_correct':int((full&ok&(roots==r)).sum())}for r in sorted(set(roots))};mac=[x['full9_correct']/x['full9_N']for x in rs.values()if x['full9_N']]
 groups={}
 for name,lo,hi in [('Holding',0,3),('In',3,6),('drawer',6,9)]:
  valid=v[:,lo:hi].all(1);groups[name]={'N':int(valid.sum()),'wrong_states':int(((p[:,lo:hi]!=y[:,lo:hi]).any(1)&valid).sum()),'FP_facts':int((p[:,lo:hi]&~y[:,lo:hi]&v[:,lo:hi]).sum()),'FN_facts':int((~p[:,lo:hi]&y[:,lo:hi]&v[:,lo:hi]).sum())}
 return {'N':len(y),'roots':len(rs),'full9_N':int(full.sum()),'full9_correct':int((full&ok).sum()),'full9_root_macro':float(np.mean(mac))if mac else None,'all_valid_correct':int(ok.sum()),'valid_facts':int(v.sum()),'correct_valid_facts':int(((p==y)&v).sum()),'groups':groups,'per_root':rs}
def paired(a,b,y,v):
 a=discrete(a)==y;b=discrete(b)==y;v=np.asarray(v,bool);full=v.all(1);sa=a.all(1);sb=b.all(1)
 def counts(a,b,mask):return {'actual_right_prediction_wrong':int((a&~b&mask).sum()),'both_wrong':int((~a&~b&mask).sum()),'actual_wrong_prediction_right':int((~a&b&mask).sum()),'both_right':int((a&b&mask).sum())}
 return {'facts':counts(a,b,v),'states':counts(sa,sb,full)}
def trajectory_summary(rr,rootN):
 result=[]
 for h in range(4):
  rows=[r for r in rr if r['slot']==h];out={'horizon':h+1,'planned_roots':rootN,'observed_roots':len(rows),'unreached_roots':rootN-len(rows),'native_status_counts':{status:sum(r['status']==status for r in rows)for status in ['success','refused','failed']}}
  if rows:
   y=np.array([r['truth']for r in rows]);v=np.array([r['valid']for r in rows]);roots=[r['root_id']for r in rows];ps={k:np.array([r[k+'_probability']for r in rows])for k in ['actual','one_step','recursive']}
   for k,p in ps.items():out[k]=metric(p,y,v,roots)
   for k in ['one_step','recursive']:out[k+'_paired']=paired(ps['actual'],ps[k],y,v);out[k+'_raw_MSE']=float(np.mean([r[k+'_raw_MSE']for r in rows]))
   a=(discrete(ps['one_step'])==y).all(1);b=(discrete(ps['recursive'])==y).all(1);full=v.all(1);out['one_step_right_recursive_wrong']=int((a&~b&full).sum());out['one_step_wrong_recursive_right']=int((~a&b&full).sum())
  result.append(out)
 return result
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['C_validation','D3'],required=True);a=ap.parse_args();d=E/'data'/('ABC'if a.phase=='C_validation'else'D3');states=read(d/'public/states.json');labels=read(d/'_private/labels.json');outcomes=read(d/'_private/outcomes.json');predictions=read(E/'results'/f'predictions_{a.phase}.json');sp=read(E/'results'/f'true_state_probabilities_{a.phase}.json');idx=sp['state_indices'];y=np.array([labels[str(i)]['y']for i in idx]);v=np.array([labels[str(i)]['v']for i in idx]);roots=[states[i]['root_id']for i in idx];rootN=len(set(roots));state_scores={};scored={};trajectory={};pairedrows=[]
 for name,prob in sp['probabilities'].items():
  prob=np.array(prob);groups={}
  for stratum in ['pre_grasp','held_open','hover_above_drawer','before_release','post_place','held_closed','other_states','skill_boundary']:
   mask=np.array([states[i]['skill_boundary']if stratum=='skill_boundary'else not states[i]['strata']if stratum=='other_states'else stratum in states[i]['strata']for i in idx]);groups[stratum]=metric(prob[mask],y[mask],v[mask],np.array(roots)[mask])
  state_scores[name]={'all_states':metric(prob,y,v,roots),'strata':groups}
 for condition,rows in predictions.items():
  scored[condition]=[]
  for r in rows:
   lab=labels[str(r['after'])];s=r|{'truth':lab['y'],'valid':lab['v'],'status':outcomes[r['id']]['status']};scored[condition].append(s)
   for j,fact in enumerate(FACTS):pairedrows.append({'condition':condition,'root':r['root_id'],'branch':r['branch'],'step':r['slot']+1,'skill':r['skill'],'status':s['status'],'fact':fact,'truth':lab['y'][j],'valid':lab['v'][j],**{k:bool(discrete(r[k+'_probability'])[j])for k in ['actual','one_step','recursive']}})
  allrows=scored[condition];branchstats={}
  for branch in ['normal','held_closed']:
   rr=[r for r in allrows if r['branch']==branch];complete={r for r in roots if sum(x['root_id']==r for x in rr)==4};branchstats[branch]={'all_available':trajectory_summary(rr,rootN),'fixed_four_step_observed_roots':sorted(complete),'fixed_four_step':trajectory_summary([r for r in rr if r['root_id']in complete],len(complete))}
  statuses={}
  for status in ['success','refused','failed']:
   rr=[r for r in allrows if r['status']==status]
   if rr:statuses[status]={k:metric(np.array([r[k+'_probability']for r in rr]),np.array([r['truth']for r in rr]),np.array([r['valid']for r in rr]),[r['root_id']for r in rr])for k in ['actual','one_step','recursive']}
  trajectory[condition]={'branches':branchstats,'by_native_status':statuses}
 # Exact same readout control under each seed's transition arms.
 selected=read(E/'configs/query_choices_frozen.json')['selected']
 for seed in[0,1]:
  qname=f'query_{selected[str(seed)]["arm"]}_seed{seed}';baseline={r['id']:r for r in scored[qname]}
  for arm in ['old_coverage','new_coverage']:
   for r in scored[f'transition_{arm}_seed{seed}']:assert np.max(np.abs(np.array(r['actual_probability'])-baseline[r['id']]['actual_probability']))<1e-5
 # Old-state retention uses same untouched C-validation states for every query head.
 retention={}
 if a.phase=='C_validation':
  cv=dict(np.load(E/'data/query_evaluator/old_C_validation.npz'));qc=torch.load(REP/'models/query_seed0.pt',map_location='cpu',weights_only=True);headstates={'original_head':qc['state']}
  for arm in ['old_coverage','new_coverage']:
   for seed in[0,1]:
    name=f'query_{arm}_seed{seed}';sel=read(E/'models'/name/'selected.json');headstates[name]=torch.load(sel['path'],map_location='cpu',weights_only=True)['state']
  torch.set_num_threads(4)
  with torch.no_grad():
   for name,state in headstates.items():q=Query();q.load_state_dict(state);p=probs(q(torch.tensor(cv['z']))).numpy();retention[name]=metric(p,cv['y'],cv['v'],cv['root_ids'])
 pair_changes=[]
 for seed in[0,1]:
  old=np.array(sp['probabilities'][f'query_old_coverage_seed{seed}']);new=np.array(sp['probabilities'][f'query_new_coverage_seed{seed}']);po=(discrete(old)==y).all(1);pn=(discrete(new)==y).all(1);full=v.all(1);pair_changes.append({'seed':seed,'full9_states':int(full.sum()),'old_wrong_new_right':int((~po&pn&full).sum()),'old_right_new_wrong':int((po&~pn&full).sum())})
 save(E/'results'/f'scored_{a.phase}.json',scored);save(E/'results'/f'metrics_{a.phase}.json',{'phase':a.phase,'evaluated_utc':now(),'independent_roots':rootN,'true_state_readout':state_scores,'trajectory_prediction':trajectory,'old_C_retention':retention,'query_paired_state_changes':pair_changes,'test_status':'D3 first sealed-model evaluation; now seen and cannot be reused as unseen'if a.phase=='D3'else'New C held-out-root validation, no selection or retuning'})
 with(R/f'paired_propositions_{a.phase}.csv').open('w')as f:w=csv.DictWriter(f,fieldnames=list(pairedrows[0]));w.writeheader();w.writerows(pairedrows)
 print('SCORED',a.phase,'roots',rootN,'paired_fact_rows',len(pairedrows),flush=True)
if __name__=='__main__':main()
