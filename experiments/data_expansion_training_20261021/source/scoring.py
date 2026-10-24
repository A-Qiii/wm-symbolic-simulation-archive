from common import *
def scope_masks(z):
 mode=z['mode'];t=z['relative_step'];origin=z['origin'];current=mode=='current';free=mode=='free';one=mode=='one_step'
 masks={'initial_all':current&(t==0),'initial_h0':current&(t==0)&(origin==0),'initial_middle':current&(t==0)&(origin>0),'observed_path':current&(origin==0),'one_step_observed_prefix':one&(origin==0),'one_step_cold_each_origin':one&(t==1),'full_free_boundaries':free&(origin==0)&(t>0),'warm_initial_middle':(mode=='warm_current')&(origin>0)}
 for h in [1,3,5,6]:
  masks[f'free_h{h}']=free&(t==h)
  masks[f'free_middle_h{h}']=free&(t==h)&(origin>0)
  masks[f'warm_middle_h{h}']=(mode=='warm_free')&(t==h)&(origin>0)
 combo=(z['cat'][:,:3]==3).sum(1)>=2;combo &=np.isin(z['cat'][:,:3],[3,6]).all(1)
 if str(z['source'][0])=='combination_test':
  for name in ['initial_all','initial_h0','initial_middle','observed_path','one_step_observed_prefix','one_step_cold_each_origin','free_h1','free_h3','free_h5','free_h6']:masks['heldout_combo_'+name]=masks[name]&combo
 return masks
def variables(z):
 p=rel(z['prob']);yy=np.c_[z['cat'][:,:3]==6,z['cat'][:,:3]==3];vv=z['cat_valid'][:,:3];joint_valid=vv.all(1)&z['cat_valid'][:,3];pred=p[:,:6]>=.5;drawer=p[:,6:9].argmax(1);ok=np.c_[(pred[:,:3]==yy[:,:3]).all(1),(pred[:,3:6]==yy[:,3:6]).all(1),drawer==z['cat'][:,3]]
 return p,yy,vv,joint_valid,ok.all(1)
def metrics(z,select):
 p,y,v,jv,correct=variables(z);ids=np.flatnonzero(select);valid_joint=select&jv;rr={'conditions':len(ids),'joint_eligible':int(valid_joint.sum()),'joint_correct':int((valid_joint&correct).sum())}
 rr['joint_accuracy']=rr['joint_correct']/rr['joint_eligible']if rr['joint_eligible']else None
 for name,off in [('Held',0),('In',3)]:
  pp=p[ids,off:off+3]>=.5;yy=y[ids,off:off+3];mask=v[ids];positive=yy&mask;negative=(~yy)&mask;tp=int((pp&positive).sum());fp=int((pp&negative).sum());n=int(positive.sum());neg=int(negative.sum())
  rr.update({name+'_positive':n,name+'_TP':tp,name+'_negative':neg,name+'_FP':fp,name+'_recall':tp/n if n else None,name+'_FP_rate':fp/neg if neg else None,name+'_precision':tp/(tp+fp)if tp+fp else None})
 dv=select&z['cat_valid'][:,3];rr['drawer_eligible']=int(dv.sum());rr['drawer_correct']=int(((p[:,6:9].argmax(1)==z['cat'][:,3])&dv).sum())
 pred=p[ids,:6];truth=y[ids].astype(float);mask=np.tile(v[ids],(1,2));rr['predicate_Brier_eligible']=int(mask.sum());rr['predicate_Brier']=float(((pred-truth)**2)[mask].mean())if mask.any()else None
 return rr
def retention(z):
 p,y,v,jv,ok=variables(z);index={(str(e),int(o),str(m),int(t)):i for i,(e,o,m,t)in enumerate(zip(z['episode_id'],z['origin'],z['mode'],z['relative_step']))};rows=[]
 for i in range(len(p)):
  t=int(z['relative_step'][i]);origin=int(z['origin'][i]);mode=str(z['mode'][i])
  if t<1:continue
  condition=None;prev=None
  if origin==0 and mode in ['current','one_step','free']:
   condition={'current':'observed_path','one_step':'one_step_observed_prefix','free':'free_h0'}[mode];pm='current'if mode=='one_step'else mode;prev=index.get((str(z['episode_id'][i]),0,pm,t-1))
  elif mode=='one_step'and t==1:
   condition='one_step_cold_middle';prev=index.get((str(z['episode_id'][i]),origin,'current',0))
  if prev is None or condition is None:continue
  target=int(z['target'][i])
  for ob in range(3):
   if ob==target or not(v[i,ob]and v[prev,ob]):continue
   for name,off in [('Held',0),('In',3)]:
    if not(y[i,off+ob]and y[prev,off+ob]):continue
    before=p[prev,off+ob]>=.5;after=p[i,off+ob]>=.5
    rows.append({'source':str(z['source'][i]),'arm':str(z['arm'][i]),'seed':int(z['seed'][i]),'family_id':str(z['family_id'][i]),'episode_id':str(z['episode_id'][i]),'origin':origin,'step':int(z['step'][i]),'condition':condition,'relation':name,'object':COLORS[ob],'both_correct':int(before and after),'previous_correct':int(before),'forgot':int(before and not after),'skill':str(z['skill'][i]),'branch_mode':str(z['branch_mode'][i])})
 return rows
def first_errors(z):
 p,y,v,jv,ok=variables(z);groups=collections.defaultdict(list);out=[]
 for i in range(len(p)):
  if z['mode'][i]in ['free','warm_free']:groups[str(z['episode_id'][i]),int(z['origin'][i]),str(z['mode'][i])].append(i)
 for (eid,origin,mode),jj in groups.items():
  jj=sorted(jj,key=lambda i:z['relative_step'][i]);eligible=[i for i in jj if jv[i]];bad=next((i for i in eligible if not ok[i]),None);i=bad if bad is not None else jj[0]
  out.append({'source':str(z['source'][i]),'arm':str(z['arm'][i]),'seed':int(z['seed'][i]),'family_id':str(z['family_id'][i]),'episode_id':eid,'origin':origin,'mode':mode,'eligible_boundaries':len(eligible),'whole_correct':bool(eligible and all(ok[k]for k in eligible)),'first_error_relative_step':int(z['relative_step'][i])if bad is not None else-1,'first_error_skill':str(z['skill'][i])if bad is not None else'none','injected_miss_step':bool(z['injected_miss_step'][i])if bad is not None else False,'stage':'initial'if bad is not None and z['relative_step'][i]==0 else'transition'if bad is not None else'none','probability':p[i].tolist()if bad is not None else[],'truth':y[i].tolist()if bad is not None else[]})
 return out
def phase():
 main=[];strata=[];ret=[];first=[];probs={};color=[]
 for file in sorted((S/'results').glob('predictions_*.npz')):
  z=dict(np.load(file));key=(str(z['source'][0]),str(z['arm'][0]),int(z['seed'][0]));probs[key]=z;meta={'source':key[0],'arm':key[1],'seed':key[2]};masks=scope_masks(z)
  for name,sel in masks.items():
   if not sel.any():continue
   main.append({**meta,'scope':name,**metrics(z,sel)})
   if name in ['initial_all','initial_middle','one_step_observed_prefix','one_step_cold_each_origin','free_h1','free_h3','free_h5']:
    for field in ['environment','state_kind','branch_mode','skill']:
     for value in np.unique(z[field][sel]):
      q=sel&(z[field]==value);strata.append({**meta,'scope':name,'factor':field,'level':str(value),**metrics(z,q)})
    p,y,v,*_=variables(z)
    for ob,c in enumerate(COLORS):
     use=sel&v[:,ob]
     for relation,off in [('Held',0),('In',3)]:
      truth=y[:,off+ob];pred=p[:,off+ob]>=.5;positive=use&truth;negative=use&~truth
      color.append({**meta,'scope':name,'object':c,'relation':relation,'positive':int(positive.sum()),'TP':int((positive&pred).sum()),'negative':int(negative.sum()),'FP':int((negative&pred).sum())})
  ret+=retention(z);first+=first_errors(z)
 aggregated=[]
 for key in sorted({(r['source'],r['arm'],r['seed'],r['condition'],r['relation'])for r in ret}):
  rr=[r for r in ret if(r['source'],r['arm'],r['seed'],r['condition'],r['relation'])==key];n=len(rr);previous=sum(r['previous_correct']for r in rr);forgot=sum(r['forgot']for r in rr)
  aggregated.append(dict(zip(['source','arm','seed','condition','relation'],key),stable_non_target_positive_intervals=n,both_correct=sum(r['both_correct']for r in rr),previous_correct=previous,forgot=forgot,forgot_given_previous_correct=forgot/previous if previous else None))
 paired=[]
 for source,seed in sorted({(k[0],k[2])for k in probs}):
  a=probs[source,'old228',seed];b=probs[source,'new868',seed]
  for name in ['episode_id','origin','mode','step','cat','cat_valid']:assert np.array_equal(a[name],b[name]),(source,seed,name)
  pa,ya,va,jv,oa=variables(a);pb,yb,vb,jvb,ob=variables(b);masks=scope_masks(a)
  for name in ['initial_all','initial_middle','one_step_observed_prefix','one_step_cold_each_origin','free_h1','free_h3','free_h5','heldout_combo_initial_all','heldout_combo_free_h3']:
   if name not in masks:continue
   sel=masks[name]&jv
   if not sel.any():continue
   fam=np.unique(a['family_id'][sel]);delta=np.array([int(ob[sel&(a['family_id']==f)].sum())-int(oa[sel&(a['family_id']==f)].sum())for f in fam]);den=np.array([int((sel&(a['family_id']==f)).sum())for f in fam]);rng=np.random.default_rng(2610219200+seed);draw=rng.integers(len(fam),size=(2000,len(fam)));dist=delta[draw].sum(1)/den[draw].sum(1)
   paired.append({'source':source,'seed':seed,'scope':name,'eligible':int(sel.sum()),'old_correct':int(oa[sel].sum()),'new_correct':int(ob[sel].sum()),'difference_new_minus_old':float(delta.sum()/den.sum()),'families':len(fam),'family_bootstrap95_low':float(np.quantile(dist,.025)),'family_bootstrap95_high':float(np.quantile(dist,.975))})
 diagnosis=[]
 for source in ['old_development','development']:
  for arm in ARMS:
   for seed in [0,1]:
    cold={(r['episode_id'],r['origin']):r for r in first if r['source']==source and r['arm']==arm and r['seed']==seed and r['mode']=='free'and r['origin']>0};warm={(r['episode_id'],r['origin']):r for r in first if r['source']==source and r['arm']==arm and r['seed']==seed and r['mode']=='warm_free'and r['origin']>0}
    if not cold:continue
    assert cold.keys()==warm.keys();keys=list(cold)
    diagnosis.append({'source':source,'arm':arm,'seed':seed,'middle_origins':len(keys),'cold_initial_first_errors':sum(cold[k]['stage']=='initial'for k in keys),'warm_initial_first_errors':sum(warm[k]['stage']=='initial'for k in keys),'cold_whole_correct':sum(cold[k]['whole_correct']for k in keys),'warm_whole_correct':sum(warm[k]['whole_correct']for k in keys),'cold_initial_error_fixed_by_public_prefix':sum(cold[k]['stage']=='initial'and warm[k]['stage']!='initial'for k in keys)})
 for name,rows in [('metrics',main),('strata',strata),('colors',color),('retention',aggregated),('retention_intervals',ret),('first_errors',first),('paired_changes',paired),('prefix_diagnostic',diagnosis)]:
  save(S/f'results/{name}.json',rows);csvwrite(S/f'results/{name}.csv',rows)
 save(S/'results/scoring_complete.json',{'utc':now(),'cohorts':sorted({k[0]for k in probs}),'models':4,'fixed_scopes':True,'no_tuning_or_fit_after_evaluation':True});print('SCORING_COMPLETE',len(main),'metric rows',flush=True)
if __name__=='__main__':phase()
