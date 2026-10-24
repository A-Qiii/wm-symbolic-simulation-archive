from common import *
import collections
def bootstrap(values):
 v=np.asarray(values,float)
 if not len(v):return [None,None]
 rng=np.random.default_rng(2610191900);return np.quantile(v[rng.integers(0,len(v),(2000,len(v)))].mean(1),[.025,.975]).tolist()
def aggregate(records,keyfn,valuefn):
 groups=collections.defaultdict(list)
 for r in records:groups[keyfn(r)].append(r)
 return groups
def main():
 sys.addaudithook(guard);start=time.monotonic();allrecords=[];current=[];effects=[];outcome=[];ds={s:dataset(s)for s in SPLITS};chains=[]
 for split,d in ds.items():
  n=len(d['root_id'])
  for seed in [0,1]:
   vis=dict(np.load(S/f'data/visual_seed{seed}_{split}.npz'));models={'visual_rssm':vis}
   for name,arm,variant in [('GT_relations','relations','truth'),('GT_physics','physics_enriched','truth'),('visual_initial_same_relation_model','relations','visual_initial')]:models[name]=dict(np.load(S/f'data/predictions_{arm}_{variant}_seed{seed}_{split}.npz'))
   models['GT_persistence']={'one_step':d['y'][:,:4],'free':np.repeat(d['y'][:,0:1],4,axis=1)}
   for e in range(n):
    for t in range(5):
     c=correct(vis['current'][e,t],d['y'][e,t]);current.append({'split':split,'seed':seed,'root_id':str(d['root_id'][e]),'episode_id':str(d['episode_id'][e]),'t':t,'h0_independent':t==0 and str(d['episode_id'][e]).endswith('_b0'),'correct':c.tolist(),'prob':vis['current'][e,t].tolist(),'truth':d['y'][e,t].tolist()})
    for model,mm in models.items():
     for mode in ['one_step','free']:
      for t in range(4):
       p=mm[mode][e,t]
       if not np.isfinite(p[:9]).all():continue
       initial=t if mode=='one_step'else 0;yc=d['y'][e,t+1];startcorrect=correct(vis['current'][e,initial],d['y'][e,initial]);base=d['y'][e,initial];changed=~correct(base,yc);cc=correct(p[:9],yc);status=int(d['status'][e,t]);successp=float(vis['success'][e,t])if model=='visual_rssm'and mode=='one_step'else float(p[9])if len(p)>9 else None;statusp=p[9:12].tolist()if len(p)>9 else None;move=vis['moved'][e,t]if model=='visual_rssm'and mode=='one_step'else p[12:15]if len(p)>12 else None;skillid=int(d['action'][e,t,:6].argmax());record={'split':split,'seed':seed,'model':model,'root_id':str(d['root_id'][e]),'episode_id':str(d['episode_id'][e]),'episode_index':e,'mode':mode,'horizon':1 if mode=='one_step'else t+1,'step':t+1,'skill':SKILLS[skillid],'actual_status':STATUS[status],'actual_controls':int(d['controls'][e,t]),'truth':yc.tolist(),'base_truth':base.tolist(),'prob':p[:9].tolist(),'correct':cc.tolist(),'changed':changed.tolist(),'visual_initial_correct':startcorrect.tolist(),'success_probability':successp,'status_probability':statusp,'moved_probability':None if move is None else move.tolist(),'moved_target':d['moved'][e,t].tolist()};allrecords.append(record)
       if successp is not None:outcome.append(record)
       if move is not None:
        target=skillid-2 if skillid in [2,3,4]else int(np.flatnonzero(d['y'][e,t,:3])[0])if skillid==5 and d['y'][e,t,:3].sum()==1 else -1
        for j in range(3):
         if d['moved_valid'][e,t,j]:effects.append({'split':split,'seed':seed,'model':model,'mode':mode,'horizon':record['horizon'],'root_id':record['root_id'],'episode_id':record['episode_id'],'step':t+1,'skill':SKILLS[skillid],'color':COLORS[j],'role':'direct_object'if j==target else'other_object','truth':int(d['moved'][e,t,j]),'prob':float(move[j]),'actual_status':STATUS[status]})
   for e in range(n):
    chain={'split':split,'seed':seed,'root_id':str(d['root_id'][e]),'episode_id':str(d['episode_id'][e]),'episode_index':e,'skills':[SKILLS[int(a[:6].argmax())]for a in d['action'][e]],'actual_statuses':[STATUS[int(x)]for x in d['status'][e]],'truth':d['y'][e].tolist(),'current_visual':vis['current'][e].tolist(),'initial_visual_all_correct':bool(correct(vis['current'][e,0],d['y'][e,0])[-1]),'models':{}}
    for model,mm in models.items():
     if not np.isfinite(mm['free'][e,:,:9]).all():continue
     c=correct(mm['free'][e,:,:9],d['y'][e,1:]);prefix=np.cumprod(c[:,-1]).astype(bool);first=int(np.flatnonzero(~c[:,-1])[0]+1)if not c[:,-1].all()else None;chain['models'][model]={'free_prob':mm['free'][e,:,:9].tolist(),'correct_by_field':c.tolist(),'prefix_all_correct':prefix.tolist(),'first_wrong_step':first}
    if chain['skills'][1:]==['lift_red_block_table','place_in_drawer','close_drawer']or chain['skills'][1:]==['lift_blue_block_table','place_in_drawer','close_drawer']or chain['skills'][1:]==['lift_pink_block_table','place_in_drawer','close_drawer']:
     j=SKILLS.index(chain['skills'][1])-2;goal=lambda y:bool(not np.any(hard(y)[:3])and hard(y)[3+j]and hard(y)[6:9].argmax()==0);chain['fixed_sequence_goal']={'object':COLORS[j],'formula':'hand_empty AND object_in_drawer AND drawer_closed','actual':goal(d['y'][e,4]),'predicted':{m:goal(v['free_prob'][-1])for m,v in chain['models'].items()},'not_new_planning_or_control':True}
    chains.append(chain)
 save(S/'data/forecast_records.json',allrecords);save(S/'data/current_records.json',current);save(S/'data/chains.json',chains);csvwrite(S/'results/movement_predictions.csv',effects)
 groups=collections.defaultdict(list)
 for r in allrecords:
  for field,j in zip(FIELDS,range(4)):
   strata=['all','initial_all_correct'if r['visual_initial_correct'][3]else'initial_all_wrong','initial_field_correct'if r['visual_initial_correct'][j]else'initial_field_wrong']
   for st in strata:
    for status in ['all',r['actual_status']]:
     for subset in ['all']+(['changed']if r['changed'][j]else[]):groups[(r['split'],r['seed'],r['model'],r['mode'],r['horizon'],st,status,field,subset)].append((r['root_id'],r['correct'][j]))
 metrics=[];rootmetrics=[]
 for key,rows in groups.items():
  rr=collections.defaultdict(list)
  for root,value in rows:rr[root].append(value)
  values=[np.mean(v)for v in rr.values()];interval=bootstrap(values);base=dict(zip(['split','seed','model','mode','horizon','stratum','actual_status','field','subset'],key));metrics.append({**base,'roots':len(rr),'n':len(rows),'correct':sum(v for _,v in rows),'root_accuracy':float(np.mean(values)),'ci_low':interval[0],'ci_high':interval[1]})
  for rid,vals in rr.items():rootmetrics.append({**base,'root_id':rid,'n':len(vals),'accuracy':float(np.mean(vals))})
 csvwrite(S/'results/relation_metrics.csv',metrics);csvwrite(S/'results/root_relation_metrics.csv',rootmetrics)
 # Positive denominators prevent all-empty/all-outside predictions from looking solved.
 positive=[]
 for key,rows in aggregate(allrecords,lambda r:(r['split'],r['seed'],r['model'],r['mode'],r['horizon']),None).items():
  out=dict(zip(['split','seed','model','mode','horizon'],key));hp=[r for r in rows if any(r['truth'][:3])];ip=[(r,j)for r in rows for j in range(3,6)if r['truth'][j]];ineg=[(r,j)for r in rows for j in range(3,6)if not r['truth'][j]];door=[r for r in rows if r['truth'][8]];positive.append({**out,'held_positive_n':len(hp),'held_identity_correct':sum(r['correct'][0]for r in hp),'in_drawer_positive_n':len(ip),'in_drawer_true_positive':sum(r['prob'][j]>=.5 for r,j in ip),'in_drawer_negative_n':len(ineg),'in_drawer_false_positive':sum(r['prob'][j]>=.5 for r,j in ineg),'drawer_open_positive_n':len(door),'drawer_open_true_positive':sum(np.argmax(r['prob'][6:9])==2 for r in door)})
 csvwrite(S/'results/positive_relations.csv',positive)
 outcomes=[]
 for key,rows in aggregate(outcome,lambda r:(r['split'],r['seed'],r['model'],r['mode'],r['horizon'],r['actual_status']),None).items():
  base=dict(zip(['split','seed','model','mode','horizon','actual_status'],key));prob=np.array([r['success_probability']for r in rows]);y=np.array([r['actual_status']=='success'for r in rows]);rootb=collections.defaultdict(list)
  for r,p,t in zip(rows,prob,y):rootb[r['root_id']].append((p-t)**2)
  outcomes.append({**base,'roots':len(rootb),'n':len(rows),'predicted_success_count':int((prob>=.5).sum()),'success_probability_mean':float(prob.mean()),'binary_success_correct':int(((prob>=.5)==y).sum()),'root_Brier':float(np.mean([np.mean(v)for v in rootb.values()])),'three_class_correct':sum(np.argmax(r['status_probability'])==STATUS.index(r['actual_status'])for r in rows)if all(r['status_probability']is not None for r in rows)else None})
 csvwrite(S/'results/outcome_metrics.csv',outcomes)
 moves=[]
 for key,rows in aggregate(effects,lambda r:(r['split'],r['seed'],r['model'],r['mode'],r['horizon'],r['role']),None).items():
  moves.append({**dict(zip(['split','seed','model','mode','horizon','role'],key)),'n':len(rows),'positive':sum(r['truth']for r in rows),'true_positive':sum(r['truth']and r['prob']>=.5 for r in rows),'negative':sum(not r['truth']for r in rows),'false_positive':sum(not r['truth']and r['prob']>=.5 for r in rows)})
 csvwrite(S/'results/movement_metrics.csv',moves)
 cm=[]
 for key,rows in aggregate(current,lambda r:(r['split'],r['seed']),None).items():
  for cohort in ['h0_independent','all_boundaries']:
   a=[r for r in rows if cohort=='all_boundaries'or r['h0_independent']]
   for j,field in enumerate(FIELDS):
    rr=collections.defaultdict(list)
    for r in a:rr[r['root_id']].append(r['correct'][j])
    vals=[np.mean(v)for v in rr.values()];inter=bootstrap(vals);cm.append({'split':key[0],'seed':key[1],'cohort':cohort,'field':field,'roots':len(rr),'n':len(a),'correct':sum(r['correct'][j]for r in a),'root_accuracy':float(np.mean(vals)),'ci_low':inter[0],'ci_high':inter[1]})
 csvwrite(S/'results/current_metrics.csv',cm)
 prefix=[]
 for split in SPLITS:
  for seed in [0,1]:
   cc=[c for c in chains if c['split']==split and c['seed']==seed]
   for model in ['visual_rssm','GT_relations','GT_physics','visual_initial_same_relation_model','GT_persistence']:
    for stratum in ['all','initial_all_correct','initial_all_wrong']:
     sub=[c for c in cc if model in c['models']and (stratum=='all'or c['initial_visual_all_correct']==(stratum=='initial_all_correct'))]
     if not sub:continue
     for h in range(4):
      rr=collections.defaultdict(list)
      for c in sub:rr[c['root_id']].append(c['models'][model]['prefix_all_correct'][h])
      vals=[np.mean(v)for v in rr.values()];prefix.append({'split':split,'seed':seed,'model':model,'stratum':stratum,'horizon':h+1,'roots':len(rr),'n':len(sub),'prefix_correct':sum(c['models'][model]['prefix_all_correct'][h]for c in sub),'root_accuracy':float(np.mean(vals))})
 csvwrite(S/'results/prefix_metrics.csv',prefix)
 # Same learned relation model: only the supplied initial relations differ.
 substitutions=[]
 for split in SPLITS:
  for seed in [0,1]:
   for mode,horizon in [('one_step',1)]+[('free',h)for h in range(1,5)]:
    for field,j in zip(FIELDS,range(4)):
     a={(r['episode_id'],r['step']):r for r in allrecords if(r['split'],r['seed'],r['model'],r['mode'],r['horizon'])==(split,seed,'GT_relations',mode,horizon)};b={(r['episode_id'],r['step']):r for r in allrecords if(r['split'],r['seed'],r['model'],r['mode'],r['horizon'])==(split,seed,'visual_initial_same_relation_model',mode,horizon)};assert a.keys()==b.keys();rr=collections.defaultdict(list)
     for key,r in a.items():rr[r['root_id']].append(int(r['correct'][j])-int(b[key]['correct'][j]))
     vals=[np.mean(v)*100 for v in rr.values()];inter=bootstrap(vals);substitutions.append({'split':split,'seed':seed,'mode':mode,'horizon':horizon,'field':field,'roots':len(rr),'n':len(a),'GT_minus_visual_initial_pp':float(np.mean(vals)),'ci_low':inter[0],'ci_high':inter[1]})
 csvwrite(S/'results/initial_state_substitution.csv',substitutions)
 # Heldout ambiguity audit, descriptive only, after model selection/evaluation.
 alias=[];train_keys={(tuple(ds['train']['initial_state'][e,t,:9]),int(ds['train']['action'][e,t,:6].argmax()))for e in range(len(ds['train']['root_id']))for t in range(4)};train_sequences={tuple(d.argmax(-1).tolist())for d in ds['train']['action'][...,:6]}
 for split,d in ds.items():
  groups=collections.defaultdict(list)
  for e in range(len(d['root_id'])):
   for t in range(4):
    key=(tuple(d['initial_state'][e,t,:9]),int(d['action'][e,t,:6].argmax()));out=(tuple(d['y'][e,t+1]),int(d['status'][e,t]));groups[key].append((out,str(d['root_id'][e]),str(d['episode_id'][e]),t))
  alias.append({'split':split,'groups':len(groups),'conflicting_groups':sum(len(set(r[0]for r in rows))>1 for rows in groups.values()),'transitions_with_train_seen_relation_action':sum(len(rows)for key,rows in groups.items()if key in train_keys),'transitions':sum(len(rows)for rows in groups.values()),'action_sequences_seen_in_train':sum(tuple(a.argmax(-1).tolist())in train_sequences for a in d['action'][...,:6]),'episodes':len(d['root_id']),'observed_next_relation_and_status_only':True})
 save(S/'results/heldout_sufficiency.json',alias)
 # Select independent test roots with explanatory contrast, not best-case-only.
 tc=[c for c in chains if c['split']=='test'];cases=[];used=set()
 predicates=[('initial_error_propagation',lambda c:not c['initial_visual_all_correct']and c['models']['visual_rssm']['first_wrong_step']is not None and c['models']['GT_relations']['first_wrong_step']is None and c['models']['visual_initial_same_relation_model']['first_wrong_step']is not None),('aligned_initial_dynamics_error',lambda c:c['initial_visual_all_correct']and c['models']['visual_rssm']['first_wrong_step']is not None),('refusal_or_failure_chain',lambda c:any(s!='success'for s in c['actual_statuses'])),('accurate_state_remaining_error',lambda c:c['models']['GT_relations']['first_wrong_step']is not None),('successful_canonical_chain',lambda c:c.get('fixed_sequence_goal',{}).get('actual',False))]
 for label,fn in predicates:
  pool=[c for c in tc if c['root_id']not in used and fn(c)]
  if pool:
   c=min(pool,key=lambda c:(c['seed'],c['episode_id']));c=dict(c);c['case_type']=label;cases.append(c);used.add(c['root_id'])
 actual_fail=[c for c in tc if 'executed_failure'in c['actual_statuses']]
 if actual_fail:
  c=dict(min(actual_fail,key=lambda c:(c['seed'],c['episode_id'])));c['case_type']='actual_physical_failure';c['shared_root_with_another_case']=c['root_id']in used;cases.append(c)
 save(S/'results/cases.json',cases);save(S/'results/analysis_complete.json',{'utc':now(),'seconds':time.monotonic()-start,'forecast_rows':len(allrecords),'relation_metric_rows':len(metrics),'current_rows':len(current),'case_count':len(cases),'case_roots':len(set(c['root_id']for c in cases)),'new_fits':0,'new_forwards':0,'C_D_samples':0});print('ANALYSIS_COMPLETE',len(allrecords),len(metrics),len(cases),flush=True)
if __name__=='__main__':main()
