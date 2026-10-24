from common import *
import collections
MODELS=['RSSM','GT_relations','RSSM_state','old_head_state','coverage_head_state','GT_physics','GT_persistence']
def mixture(p,y):
 p=np.asarray(p);y=np.asarray(y);conf=np.r_[np.where(y[:6]>0,p[:6],1-p[:6]),p[6:9][y[6:9].argmax()]];return bool(conf.min()<.8)
def rootstats(rr,key):
 roots=collections.defaultdict(list)
 for r in rr:roots[r['root_id']].append(float(r[key]))
 v=np.array([np.mean(x)for x in roots.values()]);rng=np.random.default_rng(2610192000);ci=np.quantile(v[rng.integers(len(v),size=(2000,len(v)))].mean(1),[.025,.975])if len(v)>1 else [np.nan,np.nan]
 return {'roots':len(v),'n':len(rr),'count':sum(float(r[key])for r in rr),'root_mean':float(v.mean()),'ci_low':float(ci[0]),'ci_high':float(ci[1])}
def main():
 sys.addaudithook(guard);records=[];current=[];chains=[];ds={s:dataset(s)for s in SPLITS}
 for split,d in ds.items():
  for seed in [0,1]:
   raw=np.load(S/f'data/predictions_joint_seed{seed}_{split}.npz');ix={(str(e),str(m),int(t)):i for i,(e,m,t)in enumerate(zip(raw['episode_id'],raw['mode'],raw['step']))};v={name:np.load(S/f'data/current_{name}_seed{seed}_{split}.npz')['prob']for name in ['rssm','old_object','coverage_object']};models={'RSSM':{mode:np.array([[rel_from_prob(raw['prob'][ix[str(e),'one_step'if mode=='one_step'else'free'+str(t),t]])for t in range(1,5)]for e in d['episode_id']])for mode in ['one_step','free']}}
   for name,arm,variant in [('GT_relations','relations','truth'),('RSSM_state','relations','rssm'),('old_head_state','relations','old_object'),('coverage_head_state','relations','coverage_object'),('GT_physics','physics_enriched','truth')]:models[name]=dict(np.load(S/f'data/predictions_small_{arm}_{variant}_seed{seed}_{split}.npz'))
   models['GT_persistence']={'one_step':d['y'][:,:4],'free':np.repeat(d['y'][:,0:1],4,axis=1)}
   for e,eid in enumerate(d['episode_id']):
    common={'split':split,'scene':str(d['scene'][e]),'seed':seed,'root_id':str(d['root_id'][e]),'episode_id':str(eid),'episode_index':e,'all_models_unseen':split!='train'}
    for source in v:
     for t in range(5):
      p=v[source][e,t];y=d['y'][e,t];c=correct(p,y);hh=hard(p);held=int(np.flatnonzero(y[:3])[0])if y[:3].sum()==1 else -1
      current.append({**common,'source':source,'boundary':t,'h0_independent':t==0 and str(eid).endswith('_b0'),'prob':p.tolist(),'truth':y.tolist(),'all_correct':bool(c[3]),**{f:bool(c[j])for j,f in enumerate(FIELDS[:3])},'hard_correct_but_mixed':bool(c[3]and mixture(p,y)),'true_held':held,'pred_held':int(np.flatnonzero(hh[:3])[0])if hh[:3].sum()==1 else -1 if hh[:3].sum()==0 else 3,'false_held_max':float(p[:3][y[:3]==0].max())if(y[:3]==0).any()else 0.,'true_held_probability':float(p[held])if held>=0 else None,'extra_binary':int(((hh[:6]>0)&(y[:6]==0)).sum()),'missing_binary':int(((hh[:6]==0)&(y[:6]>0)).sum())})
    chain={**common,'skills':[SKILLS[int(a[:6].argmax())]for a in d['action'][e]],'status':[STATUS[int(a)]for a in d['status'][e]],'controls':d['controls'][e].tolist(),'truth':d['y'][e].tolist(),'current':{k:v[k][e].tolist()for k in v},'models':{}}
    for model,mm in models.items():
     src='old_object'if model=='old_head_state'else'coverage_object'if model=='coverage_head_state'else'rssm';cc=correct(mm['free'][e,:,:9],d['y'][e,1:]);chain['models'][model]={'free':mm['free'][e,:,:9].tolist(),'one_step':mm['one_step'][e,:,:9].tolist(),'correct':cc.tolist(),'prefix':np.cumprod(cc[:,-1]).astype(bool).tolist(),'first_error':int(np.flatnonzero(~cc[:,-1])[0]+1)if not cc[:,-1].all()else None}
     for mode in ['one_step','free']:
      for t in range(4):
       initial=t if mode=='one_step'else 0;p=mm[mode][e,t,:9];y=d['y'][e,t+1];base=d['y'][e,initial];cc=correct(p,y);changed=~correct(base,y);start=v[src][e,initial];startok=bool(correct(start,base)[3]);mixed=mixture(start,base);hh=hard(p);before=d['y'][e,t];direct=int(np.flatnonzero(before[:3])[0])if before[:3].sum()==1 else -1;skill=SKILLS[int(d['action'][e,t,:6].argmax())];r={**common,'model':model,'mode':mode,'horizon':1 if mode=='one_step'else t+1,'step':t+1,'skill':skill,'actual_status':STATUS[int(d['status'][e,t])],'actual_controls':int(d['controls'][e,t]),'truth':y.tolist(),'prob':p.tolist(),'base_truth':base.tolist(),'correct':cc.tolist(),'changed':changed.tolist(),'all_correct':bool(cc[3]),'initial_source':src,'initial_class':'correct_mixed'if startok and mixed else'correct_confident'if startok else'wrong','initial_hard_correct':startok,'extra_binary':int(((hh[:6]>0)&(y[:6]==0)).sum()),'missing_binary':int(((hh[:6]==0)&(y[:6]>0)).sum()),'place_direct_color':direct if skill=='place_in_drawer'else -1,'post_place_target_true':bool(y[3+direct])if skill=='place_in_drawer'and direct>=0 else None,'post_place_target_pred':bool(hh[3+direct])if skill=='place_in_drawer'and direct>=0 else None,'in_TP':int(((hh[3:6]>0)&(y[3:6]>0)).sum()),'in_positive':int(y[3:6].sum()),'in_FP':int(((hh[3:6]>0)&(y[3:6]==0)).sum()),'in_negative':int((y[3:6]==0).sum())};r['success_probability']=float(raw['event_prob'][ix[str(eid),'one_step',t+1],0])if model=='RSSM'and mode=='one_step'else float(mm[mode][e,t,9])if model not in ['RSSM','GT_persistence']else None;r['predicted_status']=STATUS[int(mm[mode][e,t,9:12].argmax())]if model not in ['RSSM','GT_persistence']else None;records.append(r)
    chains.append(chain)
 save(S/'data/forecast_records.json',records);save(S/'data/current_records.json',current);save(S/'data/chains.json',chains)
 metrics=[]
 for split in SPLITS:
  for scene in ['all','A','B']:
   for seed in [0,1]:
    for model in MODELS:
     for mode,h in [('one_step',1)]+[('free',h)for h in range(1,5)]:
      rr=[r for r in records if(r['split'],r['seed'],r['model'],r['mode'],r['horizon'])==(split,seed,model,mode,h)and(scene=='all'or r['scene']==scene)]
      for st in ['all','correct_confident','correct_mixed','wrong']:
       ss=[r for r in rr if st=='all'or r['initial_class']==st]
       if not ss:continue
       for j,field in enumerate(FIELDS):
        for subset in ['all','changed']:
         aa=[{**r,'value':r['correct'][j]}for r in ss if subset=='all'or r['changed'][j]]
         if aa:metrics.append({'split':split,'scene':scene,'seed':seed,'model':model,'mode':mode,'horizon':h,'initial_class':st,'field':field,'subset':subset,**rootstats(aa,'value')})
 csvwrite(S/'results/relation_metrics.csv',metrics)
 cm=[];conf=[]
 for split in SPLITS:
  for scene in ['all','A','B']:
   for seed in [0,1]:
    for source in v:
     for cohort in ['h0_independent','all_boundaries','true_held_boundaries']:
      # Explicit cohort filter keeps h0 independent rather than tripling each root.
      rr=[r for r in current if(r['split'],r['seed'],r['source'])==(split,seed,source)and(scene=='all'or r['scene']==scene)]
      if cohort=='h0_independent':rr=[r for r in rr if r['h0_independent']]
      if cohort=='true_held_boundaries':rr=[r for r in rr if r['true_held']>=0]
      for field in FIELDS[:3]+['all_correct','hard_correct_but_mixed','extra_binary','missing_binary']:
       if rr:cm.append({'split':split,'scene':scene,'seed':seed,'source':source,'cohort':cohort,'field':field,**rootstats(rr,field)})
      if scene=='all'and cohort=='h0_independent':
       for true in [-1,0,1,2]:
        for pred in [-1,0,1,2,3]:conf.append({'split':split,'seed':seed,'source':source,'true':true,'pred':pred,'n':sum(r['true_held']==true and r['pred_held']==pred for r in rr)})
 csvwrite(S/'results/current_metrics.csv',cm);csvwrite(S/'results/h0_held_confusion.csv',conf)
 prefixes=[];extras=[];places=[]
 for split in SPLITS:
  for scene in ['all','A','B']:
   for seed in [0,1]:
    for model in MODELS:
     cc=[c for c in chains if(c['split'],c['seed'])==(split,seed)and(scene=='all'or c['scene']==scene)]
     for h in range(4):prefixes.append({'split':split,'scene':scene,'seed':seed,'model':model,'horizon':h+1,**rootstats([{**c,'value':c['models'][model]['prefix'][h]}for c in cc],'value')})
     for mode,h in [('one_step',1)]+[('free',h)for h in range(1,5)]:
      rr=[r for r in records if(r['split'],r['seed'],r['model'],r['mode'],r['horizon'])==(split,seed,model,mode,h)and(scene=='all'or r['scene']==scene)]
      for field in ['extra_binary','missing_binary']:extras.append({'split':split,'scene':scene,'seed':seed,'model':model,'mode':mode,'horizon':h,'field':field,**rootstats(rr,field)})
      aa=[r for r in rr if r['skill']=='place_in_drawer']
      if aa:
       direct=[r for r in aa if r['place_direct_color']>=0];places.append({'split':split,'scene':scene,'seed':seed,'model':model,'mode':mode,'horizon':h,'place_transitions':len(aa),'executed_place':sum(r['actual_controls']>0 for r in aa),'direct_target_records':len(direct),'direct_target_true_positive':sum(r['post_place_target_true']and r['post_place_target_pred']for r in direct),'direct_target_positive':sum(r['post_place_target_true']for r in direct),'direct_target_false_positive':sum(not r['post_place_target_true']and r['post_place_target_pred']for r in direct),'in_TP':sum(r['in_TP']for r in aa),'in_positive':sum(r['in_positive']for r in aa),'in_FP':sum(r['in_FP']for r in aa),'in_negative':sum(r['in_negative']for r in aa),'all_membership_correct':sum(r['correct'][1]for r in aa)})
 csvwrite(S/'results/prefix_metrics.csv',prefixes);csvwrite(S/'results/extra_relations.csv',extras);csvwrite(S/'results/post_place_metrics.csv',places)
 old=dict(np.load(D/'data/dataset_train.npz'));trainmap=collections.defaultdict(list)
 for e in range(len(old['episode_id'])):
  for t in range(4):trainmap[(tuple(old['y'][e,t]),int(old['action'][e,t,:6].argmax()))].append({'episode_id':str(old['episode_id'][e]),'root_id':str(old['root_id'][e]),'step':t,'after':old['y'][e,t+1].tolist(),'status':STATUS[int(old['status'][e,t])],'physical_before':old['physical'][e,t].tolist()})
 aliases=[];conflicts=[]
 for split,d in ds.items():
  groups=collections.defaultdict(list);seen=0;different=0;different_relation=0
  for e,eid in enumerate(d['episode_id']):
   for t in range(4):
    key=(tuple(d['y'][e,t]),int(d['action'][e,t,:6].argmax()));out=(tuple(d['y'][e,t+1]),STATUS[int(d['status'][e,t])]);groups[key].append(out);oldout={(tuple(r['after']),r['status'])for r in trainmap[key]};seen+=bool(oldout);is_diff=bool(oldout and out not in oldout);different+=is_diff;different_relation+=bool(oldout and out[0]not in {o[0]for o in oldout})
    if is_diff:conflicts.append({'split':split,'scene':str(d['scene'][e]),'root_id':str(d['root_id'][e]),'episode_id':str(eid),'step':t,'skill':SKILLS[key[1]],'before_relation':list(key[0]),'after_relation':list(out[0]),'status':out[1],'old_examples':trainmap[key][:2],'physical_before':d['physical'][e,t].tolist(),'relation_conflict':out[0]not in {o[0]for o in oldout}})
  aliases.append({'split':split,'groups':len(groups),'within_split_conflicting_groups':sum(len(set(v))>1 for v in groups.values()),'transitions':len(d['episode_id'])*4,'train_seen_relation_action':seen,'different_observed_relation_or_status_from_train':different,'different_relation_from_train':different_relation})
 save(S/'results/state_sufficiency.json',aliases);save(S/'results/state_conflicts.json',conflicts)
 # Cases from the genuinely unused-by-all-models roots first; all roots unique.
 tc=[c for c in chains if c['split']!='train'];cases=[];used=set()
 specs=[('initial_identity_propagates',lambda c:not correct(c['current']['rssm'][0],np.array(c['truth'][0]))[0]and c['models']['RSSM']['first_error']is not None),('separate_head_composition_repairs',lambda c:c['models']['RSSM']['first_error']is not None and c['models']['coverage_head_state']['first_error']is None),('hard_correct_soft_mixed',lambda c:correct(c['current']['rssm'][0],np.array(c['truth'][0]))[3]and mixture(c['current']['rssm'][0],c['truth'][0])and c['models']['RSSM']['first_error']is not None),('accurate_state_generalization_failure',lambda c:c['models']['GT_relations']['first_error']is not None),('coverage_head_regression',lambda c:c['models']['old_head_state']['first_error']is None and c['models']['coverage_head_state']['first_error']is not None),('other_scene_identity_propagation',lambda c:not correct(c['current']['rssm'][0],np.array(c['truth'][0]))[0]and c['models']['RSSM']['first_error']is not None)]
 for typ,pred in specs:
  pool=[c for c in tc if c['root_id']not in used and pred(c)]
  if typ=='other_scene_identity_propagation'and cases:pool=[c for c in pool if c['scene']!=cases[0]['scene']]
  if pool:c=dict(min(pool,key=lambda c:(c['seed'],c['episode_id'])));c['case_type']=typ;cases.append(c);used.add(c['root_id'])
 pool=[c for c in tc if c['scene']=='A'and c['root_id']not in used and any(not correct(c['current']['coverage_object'][t],np.array(c['truth'][t]))[0]and correct(c['current']['old_object'][t],np.array(c['truth'][t]))[0]for t in range(5))]
 if pool:
  c=dict(min(pool,key=lambda c:(c['seed'],c['episode_id'])));c['case_type']='intermediate_head_regression';cases.append(c);used.add(c['root_id'])
 save(S/'results/cases.json',cases);save(S/'results/analysis_complete.json',{'utc':now(),'forecast_rows':len(records),'current_rows':len(current),'metrics_rows':len(metrics),'cases':len(cases),'unique_case_roots':len(used),'state_conflicts':len(conflicts),'fits':0,'forwards':0});print('ANALYSIS_COMPLETE',len(records),len(metrics),aliases,[(c['case_type'],c['episode_id'])for c in cases],flush=True)
if __name__=='__main__':main()
