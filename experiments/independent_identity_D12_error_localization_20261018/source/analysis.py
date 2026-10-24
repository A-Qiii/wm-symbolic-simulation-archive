"""Only stdlib operations on saved JSON probabilities. No model/data collection."""
from pathlib import Path
import json, hashlib, csv, itertools, collections, datetime, shutil
P=Path('/home/robot/research/projects/wm_symbolic')
D=P/'experiments/independent_identity_D12_20261017'
E=P/'experiments/independent_identity_D12_error_localization_20261018'
R=P/'reports/independent_identity_D12_error_localization_20261018'
FACTS=['Holding(red)','Holding(blue)','Holding(pink)','In(red,drawer)','In(blue,drawer)','In(pink,drawer)','Closed','Open','Intermediate']
GROUPS={'Holding':list(range(3)),'In':list(range(3,6)),'drawer':list(range(6,9)),'all9':list(range(9))}
BRANCHES=['normal','reverse_then_lift','place_then_other_lift'];ARMS=['original','object_control','object_coverage'];PATHS=['actual','one_step','recursive']
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def emit(name,rows):
 save(E/'results'/f'{name}.json',rows)
 with (R/f'{name}.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows({k:json.dumps(v,ensure_ascii=False)if isinstance(v,(list,dict))else v for k,v in row.items()}for row in rows)
def classify(p):return [x>=.5 for x in p[:6]]+[j==max(range(3),key=lambda i:p[6+i])for j in range(3)]
def filter_scope(rows,scope):return [r for r in rows if scope=='all'or r['pair_group']==scope]
def main():
 assert not E.exists()and not R.exists(),'Do not overwrite any prior appendix'
 original_manifest=read(D/'configs/delivery_manifest.json')
 for p,h in original_manifest['sha256'].items():assert sha(p)==h,p
 R.mkdir(parents=True);(E/'source').mkdir(parents=True)
 shutil.copy2(__file__,E/'source/analysis.py')
 names=['results/predictions.json','results/collection.json','data/combo/public/pairs.json','data/combo/_private/labels.json','configs/roots.json','results/root_contexts.json','results/chain_root_results.json','results/stage_metrics.json','results/primary.json','configs/delivery_manifest.json','results/completion_audit.json']
 save(E/'configs/input_manifest.json',{'utc':now(),'purpose':'post hoc descriptive error localization; no change to frozen scoring or acceptance','sha256':{str(D/n):sha(D/n)for n in names},'source_sha256':sha(__file__),'model_forward_calls':0,'training_updates':0,'collection_calls':0,'threshold_changes':0})
 preds=read(D/'results/predictions.json');labs=read(D/'data/combo/_private/labels.json');roots={r['root_id']:r for r in read(D/'configs/roots.json')};eligible={r['root_id']for r in read(D/'results/root_contexts.json')if r['context_matched']}
 slots={(r['root_id'],r['branch'],s['slot']):s for r in read(D/'results/collection.json')for s in r['slots']}
 base={(p['seed'],p['query_arm'],p['root_id'],p['branch'],p['slot']):p for p in preds if p['world_model']=='latent'}
 boundaries=[];lookup={};conditions=set()
 for p in preds:
  assert p['actual_probability']==base[(p['seed'],p['query_arm'],p['root_id'],p['branch'],p['slot'])]['actual_probability']
  for path in PATHS:
   if path=='actual'and p['world_model']!='latent':continue
   wm='observed'if path=='actual'else p['world_model'];prob=p[path+'_probability'];y=labs[str(p['after'])]['y'];v=labs[str(p['after'])]['v'];c=classify(prob);cond=(p['seed'],wm,p['query_arm'],path);conditions.add(cond)
   row={k:p[k]for k in ['root_id','target','other_target','branch','slot','skill','after']};row.update(seed=p['seed'],world_model=wm,query_arm=p['query_arm'],path=path,pair_group=roots[p['root_id']]['pair_group'],step=p['slot']+1,eligible_root=p['root_id']in eligible,execution_status=slots[(p['root_id'],p['branch'],p['slot'])]['status'],truth=y,valid=v,probability=prob,classification=c,wrong_facts=[FACTS[j]for j in range(9)if v[j]and y[j]!=c[j]],false_positive=[FACTS[j]for j in range(9)if v[j]and not y[j]and c[j]],false_negative=[FACTS[j]for j in range(9)if v[j]and y[j]and not c[j]])
   for group,idx in GROUPS.items():row[group+'_error']=any(y[j]!=c[j]for j in idx)if all(v[j]for j in idx)else None
   boundaries.append(row);lookup[(*cond,p['root_id'],p['branch'],p['slot'])]=row
 assert len(conditions)==30 and len(boundaries)==4050
 emit('boundary_errors',boundaries)
 stages=[];first=[];chain_summary=[];first_summary=[];objects=[];joint_roots=[];joint_summary=[]
 for cond in sorted(conditions):
  seed,wm,arm,path=cond; meta=dict(seed=seed,world_model=wm,query_arm=arm,path=path);rr=[r for r in boundaries if all(r[k]==v for k,v in meta.items())]
  for scope in ['all','seen','reverse']:
   scoped=filter_scope(rr,scope);root_ids=[rid for rid,r in roots.items()if scope=='all'or r['pair_group']==scope]
   for cohort in ['all_observed','eligible_roots']:
    cc=[r for r in scoped if cohort=='all_observed'or r['eligible_root']]
    for branch,step in itertools.product(BRANCHES,range(1,5)):
     cells=[r for r in cc if r['branch']==branch and r['step']==step]
     row=dict(**meta,scope=scope,cohort=cohort,branch=branch,step=step,N=len(cells),root_N=len({r['root_id']for r in cells}),failed_calls=sum(r['execution_status']=='failed'for r in cells))
     for g,idx in GROUPS.items():
      row[g+'_valid_N']=sum(r[g+'_error']is not None for r in cells);row[g+'_wrong']=sum(r[g+'_error']is True for r in cells)
      row[g+'_FP_facts']=sum(sum(r['valid'][j]and not r['truth'][j]and r['classification'][j]for j in idx)for r in cells);row[g+'_FN_facts']=sum(sum(r['valid'][j]and r['truth'][j]and not r['classification'][j]for j in idx)for r in cells)
     stages.append(row)
   scoped_first=[]
   for rid,branch in itertools.product(root_ids,BRANCHES):
    cells=sorted([r for r in scoped if r['root_id']==rid and r['branch']==branch],key=lambda r:r['step']);bad=next((r for r in cells if r['all9_error']),None);complete=len(cells)==4 and all(r['all9_error']is not None for r in cells)
    row=dict(**meta,scope=scope,root_id=rid,pair_group=roots[rid]['pair_group'],branch=branch,eligible_root=rid in eligible,observed_boundaries=len(cells),complete_valid_chain=complete,all4_correct=not any(r['all9_error']for r in cells)if complete else None,first_error_step=bad['step']if bad else None,first_error_action=bad['skill']if bad else None,first_error_execution_status=bad['execution_status']if bad else None,first_error_facts=bad['wrong_facts']if bad else [],first_error_FP=bad['false_positive']if bad else [],first_error_FN=bad['false_negative']if bad else [])
    scoped_first.append(row)
    if scope=='all':
     first.append(row)
     if bad:
      for fact in bad['wrong_facts']:
       obj=fact.split('(')[1].split(',')[0].rstrip(')')if '('in fact else None
       role='query_c'if obj==bad['target']else'held_target_d'if obj==bad['other_target']else'other_color'if obj else'drawer_state'
       objects.append(dict(**meta,root_id=rid,pair_group=roots[rid]['pair_group'],branch=branch,eligible_root=rid in eligible,step=bad['step'],action=bad['skill'],fact=fact,object=obj,object_role=role,error_type='FP'if fact in bad['false_positive']else'FN'))
   for branch in BRANCHES:
    ff=[r for r in scoped_first if r['branch']==branch];chain_summary.append(dict(**meta,scope=scope,branch=branch,planned_roots=len(ff),complete_roots=sum(r['complete_valid_chain']for r in ff),all4_correct=sum(r['all4_correct']is True for r in ff)))
    for cohort in ['all_observed','eligible_roots']:
     ff2=[r for r in ff if cohort=='all_observed'or r['eligible_root']]
     patterns=collections.Counter((r['first_error_step'],r['first_error_action'],tuple(r['first_error_facts']),tuple(r['first_error_FP']),tuple(r['first_error_FN']))for r in ff2)
     for (step,action,facts,fp,fn),count in patterns.items():first_summary.append(dict(**meta,scope=scope,cohort=cohort,branch=branch,step=step,action=action,facts=list(facts),FP=list(fp),FN=list(fn),roots=count,cohort_roots=len(ff2)))
   jrows=[]
   for rid in root_ids:
    if rid not in eligible:continue
    cells=[r for r in scoped if r['root_id']==rid and r['branch']in BRANCHES[1:]];assert len(cells)==8
    errors={g:any(r[g+'_error']for r in cells)for g in GROUPS};row=dict(**meta,scope=scope,root_id=rid,wrong_groups=[g for g in ['Holding','In','drawer']if errors[g]],wrong_boundaries=sum(r['all9_error']for r in cells),total_boundaries=8,**{g+'_any_error':x for g,x in errors.items()});jrows.append(row)
    if scope=='all':joint_roots.append(row)
   for groups,count in collections.Counter(tuple(r['wrong_groups'])for r in jrows).items():joint_summary.append(dict(**meta,scope=scope,wrong_groups=list(groups),roots=count,eligible_roots=len(jrows)))
 emit('step_errors',stages);emit('first_errors',first);emit('first_error_summary',first_summary);emit('first_error_objects',objects);emit('chain_success',chain_summary);emit('joint_root_errors',joint_roots);emit('joint_error_patterns',joint_summary)
 paircells=[]
 for p in preds:
  y=labs[str(p['after'])]['y'];v=labs[str(p['after'])]['v'];actual=classify(p['actual_probability'])
  for path in ['one_step','recursive']:
   predicted=classify(p[path+'_probability'])
   for group,idx in GROUPS.items():
    ae=any(actual[j]!=y[j]for j in idx)if all(v[j]for j in idx)else None;pe=any(predicted[j]!=y[j]for j in idx)if all(v[j]for j in idx)else None
    paircells.append(dict(seed=p['seed'],world_model=p['world_model'],query_arm=p['query_arm'],path=path,root_id=p['root_id'],pair_group=roots[p['root_id']]['pair_group'],eligible_root=p['root_id']in eligible,branch=p['branch'],step=p['slot']+1,skill=p['skill'],after=p['after'],group=group,actual_error=ae,predicted_error=pe,category=('both_wrong'if ae and pe else'actual_only_wrong'if ae else'predicted_only_wrong'if pe else'both_correct')if ae is not None and pe is not None else'unknown'))
 emit('actual_prediction_pairs',paircells);pairsummary=[]
 for seed,wm,arm,path,group,scope,cohort,branch,step in itertools.product([0,1],['mixed','latent'],ARMS,['one_step','recursive'],GROUPS,['all','seen','reverse'],['all_observed','eligible_roots'],['all']+BRANCHES,[0,1,2,3,4]):
  cc=[r for r in paircells if r['seed']==seed and r['world_model']==wm and r['query_arm']==arm and r['path']==path and r['group']==group and(scope=='all'or r['pair_group']==scope)and(cohort=='all_observed'or r['eligible_root'])and(branch=='all'or r['branch']==branch)and(step==0 or r['step']==step)]
  ct=collections.Counter(r['category']for r in cc);pairsummary.append(dict(seed=seed,world_model=wm,query_arm=arm,path=path,group=group,scope=scope,cohort=cohort,branch=branch,step=step,N=len(cc),root_N=len({r['root_id']for r in cc}),**{k:ct[k]for k in ['both_wrong','actual_only_wrong','predicted_only_wrong','both_correct','unknown']}))
 emit('actual_prediction_summary',pairsummary)
 # Three minimal probability contrasts, all using the same fixed candidate condition.
 examples=[('empty_close', 'D_identity_blue_red_00','reverse_then_lift',1),('actual_In_miss','D_identity_blue_red_00','place_then_other_lift',3)]
 candidates=sorted([r for r in paircells if r['seed']==0 and r['world_model']=='latent'and r['query_arm']=='object_coverage'and r['path']=='recursive'and r['group']=='all9'and r['eligible_root']and r['category']=='both_wrong'],key=lambda r:(r['root_id'],BRANCHES.index(r['branch']),r['step']))
 if candidates:
  x=candidates[0];examples.append(('both_wrong_posthoc_first_lex',x['root_id'],x['branch'],x['step']-1))
 output=[]
 for title,rid,branch,slot in examples:
  p=next(p for p in preds if p['seed']==0 and p['world_model']=='latent'and p['query_arm']=='object_coverage'and p['root_id']==rid and p['branch']==branch and p['slot']==slot)
  for j,f in enumerate(FACTS):output.append(dict(case=title,root_id=rid,branch=branch,step=slot+1,skill=p['skill'],fact=f,truth=labs[str(p['after'])]['y'][j],valid=labs[str(p['after'])]['v'][j],**{path+'_probability':p[path+'_probability'][j]for path in PATHS}))
 emit('probability_examples',output)
 # Cross-check against pre-existing independently verified scores.
 checks=0
 for s in read(D/'results/stage_metrics.json'):
  if s['path']=='actual'and s['world_model']=='mixed':continue
  wm='observed'if s['path']=='actual'else s['world_model']
  t=next(t for t in stages if t['seed']==s['seed']and t['world_model']==wm and t['query_arm']==s['query_arm']and t['path']==s['path']and t['scope']==s['scope']and t['cohort']=='all_observed'and t['branch']==s['branch']and t['step']==s['step'])
  for g,orig in [('Holding','Holding'),('In','In'),('drawer','drawer'),('all9','full9')]:assert t[g+'_wrong']==s[orig+'_N']-s[orig+'_correct'];checks+=1
 save(E/'results/analysis_audit.json',{'utc':now(),'saved_prediction_rows':len(preds),'unique_conditions_after_deduplicating_actual':len(conditions),'boundary_condition_rows':len(boundaries),'existing_stage_metric_checks':checks,'group_count_is_any_error_not_sum_of_fact_errors':True,'actual_not_doubled_across_world_models':True,'model_forward_calls':0,'training_updates':0,'collection_calls':0,'posthoc_example_not_model_selection':True})
 print('LOCALIZATION_ANALYSIS_DONE',len(boundaries),'boundary condition rows',len(pairsummary),'paired summaries',flush=True)
if __name__=='__main__':main()
