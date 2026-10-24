from config import *
import collections,itertools,csv
def output(name,rows):
 save(E/'results'/f'{name}.json',rows)
 if rows:
  with(R/f'{name}.csv').open('w')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def confusion(y,p,v):
 y,p,v=map(lambda x:np.asarray(x,bool),(y,p,v));return {'TP':int((v&y&p).sum()),'FN':int((v&y&~p).sum()),'FP':int((v&~y&p).sum()),'TN':int((v&~y&~p).sum()),'unknown':int((~v).sum())}
def context_check(negative,positive,c,d):
 if negative is None or positive is None:return ['missing_terminal']
 ix=[c+3,d+3,d,7]
 if not all(negative['v'][i]and positive['v'][i]for i in ix):return ['required_truth_unknown']
 reasons=[]
 if negative['y'][c+3]or not positive['y'][c+3]:reasons.append('In_c_not_false_true')
 if negative['y'][d+3]or positive['y'][d+3]:reasons.append('held_d_In_not_false')
 if not(negative['y'][d]and positive['y'][d]):reasons.append('d_not_held_both')
 if not(negative['y'][7]and positive['y'][7]):reasons.append('drawer_not_open_both')
 return reasons
def pair_score(negative,positive,c,d):
 n,p=discrete(negative),discrete(positive);return bool(not n[c+3]and p[c+3]and not n[d+3]and not p[d+3])
def full9(prob,lab):return bool(np.array_equal(discrete(prob),lab['y']))if all(lab['v'])else None
def main():
 verify_frozen();assert not(E/'results/primary.json').exists();roots=read(E/'configs/roots.json');rl={r['root_id']:r for r in roots};pairs=read(E/'data/combo/public/pairs.json');labels=read(E/'data/combo/_private/labels.json');pred=read(E/'results/predictions.json');lookup={(p['seed'],p['world_model'],p['query_arm'],p['root_id'],p['branch'],p['slot']):p for p in pred};pl={(r['root_id'],r['branch'],r['slot']):r for r in pairs};contexts=[];rootpair=[];chain=[]
 for root in roots:
  rid=root['root_id'];c=COLORS.index(root['target']);d=COLORS.index(root['other_target']);term=[pl.get((rid,b,3))for b in BRANCHES[1:]];labs=[labels[str(x['after'])]if x else None for x in term];reasons=context_check(*labs,c,d);meta={k:root[k]for k in ['root_id','target','other_target','pair_group']};contexts.append(meta|{'both_terminal_observed':all(x is not None for x in term),'context_matched':not reasons,'exclusion_reasons':reasons,'negative_truth':labs[0]['y']if labs[0]else None,'negative_valid':labs[0]['v']if labs[0]else None,'positive_truth':labs[1]['y']if labs[1]else None,'positive_valid':labs[1]['v']if labs[1]else None})
  for seed,wm,arm,path in itertools.product([0,1],['mixed','latent'],ARMS,METHODS):
   records=[lookup.get((seed,wm,arm,rid,b,3))for b in BRANCHES[1:]];row=meta|{'seed':seed,'world_model':wm,'query_arm':arm,'path':path,'eligible':not reasons,'In_pair_correct':None,'Holding_d_both_correct':None,'In_pair_and_Holding_d_correct':None,'Holding3_both_correct':None,'full9_terminal_pair_correct':None,'full9_both_chains_correct':None,'negative_In_c_probability':None,'positive_In_c_probability':None,'negative_In_d_probability':None,'positive_In_d_probability':None}
   if not reasons:
    pp=[r[path+'_probability']for r in records];pr=discrete(pp);ok=pair_score(*pp,c,d);hc=bool(pr[:,d].all());h3=bool((pr[:,:3]==np.array([l['y'][:3]for l in labs])).all())if all(all(l['v'][:3])for l in labs)else None;f=[full9(prob,lab)for prob,lab in zip(pp,labs)];row.update(In_pair_correct=ok,Holding_d_both_correct=hc,In_pair_and_Holding_d_correct=ok and hc,Holding3_both_correct=h3,full9_terminal_pair_correct=all(f)if all(x is not None for x in f)else None,negative_In_c_probability=float(pp[0][c+3]),positive_In_c_probability=float(pp[1][c+3]),negative_In_d_probability=float(pp[0][d+3]),positive_In_d_probability=float(pp[1][d+3]))
    checks=[]
    for b in BRANCHES[1:]:
     for j in range(4):
      rr=lookup.get((seed,wm,arm,rid,b,j));checks.append(full9(rr[path+'_probability'],labels[str(rr['after'])])if rr else None)
    row['full9_both_chains_correct']=all(checks)if all(x is not None for x in checks)else None
   rootpair.append(row)
   for b in BRANCHES:
    ok=[];wrongfacts=[]
    for j in range(4):
     r=lookup.get((seed,wm,arm,rid,b,j));lab=labels[str(r['after'])]if r else None;ok.append(full9(r[path+'_probability'],lab)if r else None);wrongfacts.append([FACTS[k]for k in range(9)if lab['v'][k]and discrete(r[path+'_probability'])[k]!=lab['y'][k]]if r else[])
    first=next((j+1 for j,x in enumerate(ok)if x is False),None);chain.append(meta|{'seed':seed,'world_model':wm,'query_arm':arm,'path':path,'branch':b,'observed_boundaries':sum((rid,b,j)in pl for j in range(4)),'step_correct':ok,'all4_correct':all(ok)if all(x is not None for x in ok)else None,'first_wrong_step':first,'first_wrong_facts':wrongfacts[first-1]if first else[],'first_unscorable_step':next((j+1 for j,x in enumerate(ok)if x is None),None)})
 output('root_contexts',contexts);output('root_pair_results',rootpair);output('chain_root_results',chain);primary=[];paired=[];chainmetrics=[];stages=[];facts=[];held=[]
 for scope,seed,wm,path in itertools.product(['all','seen','reverse'],[0,1],['mixed','latent'],METHODS):
  planned=[r for r in roots if scope=='all'or r['pair_group']==scope];context=[r for r in contexts if scope=='all'or r['pair_group']==scope];common={r['root_id']for r in context if r['context_matched']};meta=dict(scope=scope,seed=seed,world_model=wm,path=path)
  for arm in ARMS:
   rr=[r for r in rootpair if r['seed']==seed and r['world_model']==wm and r['path']==path and r['query_arm']==arm and r['root_id']in{p['root_id']for p in planned}];row=meta|{'query_arm':arm,'planned_roots':len(planned),'both_terminal_observed':sum(r['both_terminal_observed']for r in context),'eligible_roots':len(common),'ineligible_roots':len(planned)-len(common)}
   for metric in ['In_pair_correct','Holding_d_both_correct','In_pair_and_Holding_d_correct','Holding3_both_correct','full9_terminal_pair_correct','full9_both_chains_correct']:
    row[metric+'_n']=sum(r[metric]is True for r in rr);row[metric+'_N']=sum(r[metric]is not None for r in rr)
   row['planned_primary_success_rate']=row['In_pair_correct_n']/len(planned);primary.append(row)
  for base in ['object_control','original']:
   ll={r['root_id']:r for r in rootpair if r['seed']==seed and r['world_model']==wm and r['path']==path and r['query_arm']==base};hh={r['root_id']:r for r in rootpair if r['seed']==seed and r['world_model']==wm and r['path']==path and r['query_arm']=='object_coverage'}
   for metric in ['In_pair_correct','In_pair_and_Holding_d_correct','full9_terminal_pair_correct','full9_both_chains_correct']:
    eligible=[rid for rid in sorted(common)if ll[rid][metric]is not None and hh[rid][metric]is not None];dd=[int(hh[rid][metric])-int(ll[rid][metric])for rid in eligible];paired.append(meta|{'baseline':base,'metric':metric,'N':len(dd),'gains':dd.count(1),'losses':dd.count(-1),'ties':dd.count(0),'root_ids':eligible,'differences':dd})
  for arm,b in itertools.product(ARMS,BRANCHES):
   cc=[r for r in chain if r['seed']==seed and r['world_model']==wm and r['path']==path and r['query_arm']==arm and r['branch']==b and(scope=='all'or r['pair_group']==scope)];chainmetrics.append(meta|{'query_arm':arm,'branch':b,'planned_roots':len(planned),'complete_observed':sum(r['observed_boundaries']==4 for r in cc),'full9_valid_complete':sum(r['all4_correct']is not None for r in cc),'all4_correct':sum(r['all4_correct']is True for r in cc)})
   for j in range(4):
    rr=[r for r in pred if r['seed']==seed and r['world_model']==wm and r['query_arm']==arm and r['branch']==b and r['slot']==j and(scope=='all'or rl[r['root_id']]['pair_group']==scope)]
    if not rr:continue
    y=np.array([labels[str(r['after'])]['y']for r in rr],bool);v=np.array([labels[str(r['after'])]['v']for r in rr],bool);pp=discrete([r[path+'_probability']for r in rr]);sm=meta|{'query_arm':arm,'branch':b,'step':j+1};row=sm|{'observed_N':len(rr)}
    for group,ix in [('full9',range(9)),('Holding',range(3)),('In',range(3,6)),('drawer',range(6,9))]:eligible=v[:,ix].all(1);row[group+'_N']=int(eligible.sum());row[group+'_correct']=int((eligible&(pp[:,ix]==y[:,ix]).all(1)).sum())
    stages.append(row)
    for k,f in enumerate(FACTS):facts.append(sm|{'fact':f,**confusion(y[:,k],pp[:,k],v[:,k])})
    mask=y[:,:3]&v[:,:3];held.append(sm|{'self_held_cells':int(mask.sum()),**confusion(y[:,3:6][mask],pp[:,3:6][mask],v[:,3:6][mask])})
 output('primary',primary);output('paired_effects',paired);output('chain_metrics',chainmetrics);output('stage_metrics',stages);output('fact_metrics',facts);output('self_held_In',held)
 # Fixed original Holding/drawer preserved; actual probabilities identical across world models.
 for r in pred:
  old=lookup[(r['seed'],r['world_model'],'original',r['root_id'],r['branch'],r['slot'])]
  for path in METHODS:assert all(r[path+'_probability'][j]==old[path+'_probability'][j]for j in [0,1,2,6,7,8])
  if r['world_model']=='mixed':assert r['actual_probability']==lookup[(r['seed'],'latent',r['query_arm'],r['root_id'],r['branch'],r['slot'])]['actual_probability']
 save(E/'results/scoring_audit.json',{'utc':now(),'root_unit':12,'seen_roots':6,'reverse_roots':6,'branches_and_seeds_not_additional_samples':True,'primary_eligibility_independent_of_predictions':True,'all_observed_failed_boundaries_kept':True,'Holding_drawer_bitwise_preserved':True,'no_new_gate_or_adoption':True,'old_C_retention_evidence_unchanged':True});print('IDENTITY_SCORED',len(primary),len(paired),len(facts),flush=True)
if __name__=='__main__':main()
