from common import *
def bootstrap(a):
 a=np.asarray(a,float);r=np.random.default_rng(261019551);b=a[r.integers(len(a),size=(2000,len(a)))].mean(1);return {'difference':float(a.mean()),'lo':float(np.quantile(b,.025)),'hi':float(np.quantile(b,.975)),'families':len(a)}
def analyze(path):
 z=dict(np.load(path));p=rel(z['prob']);y=truth(z['cat']);ok=correct(p,y)[:,-1]&z['cat_valid'][:,:4].all(1);full=(z['prob'][:,:24].reshape(-1,3,8).argmax(-1)==z['cat'][:,:3]).all(1)&(z['prob'][:,24:27].argmax(-1)==z['cat'][:,3])&z['cat_valid'][:,:4].all(1);rr=[];ff=[];events=[];first=[]
 for rid in sorted(set(z['root_id'])):
  ids=np.where(z['root_id']==rid)[0];i=ids[0];ix={(int(z['branch'][j]),str(z['mode'][j]),int(z['step'][j])):j for j in ids};get=lambda b,m,t:ix[b,m,t];role=z['kind'][i]=='role';branches=[1,2]if role else[0,1,2];length=6 if role else 4;meta={k:str(z[k][i])for k in ['arm','split','kind','root_id','family_id']};meta['seed']=int(z['seed'][i]);both=lambda keys:all(ok[get(b,m,t)]for b in branches for m,t in keys)
  rec={**meta,'valid_pair':bool(z['valid_pair'][i]),'h0_relation':int(both([('current',0)])),'h0_full':int(all(full[get(b,'current',0)]for b in branches)),'h0_held':int(all(((p[get(b,'current',0),:3]>=.5)==y[get(b,'current',0),:3]).all()for b in branches)),'normal_whole':int(all(ok[get(1 if role else 0,'current',t)]for t in range(length+1))),'normal_whole_full':int(all(full[get(1 if role else 0,'current',t)]for t in range(length+1))),'all_branches_whole':int(both([('current',t)for t in range(length+1)])),'h2_pair':int(both([('current',2)])),'terminal':int(both([('suffix',length)]))}
  if not role:
   rec['h0_full']=int(full[get(0,'current',0)]);rec['h0_relation']=int(ok[get(0,'current',0)])
  if role:
   h=COLORS.index(str(z['held_color'][i]));rec.update({'joint3_actual':int(both([('current',2)]+[('suffix',t)for t in [3,4,5]])),'joint4_actual':int(both([('current',2)]+[('suffix',t)for t in [3,4,5,6]])),'disturbed_whole':int(all(ok[get(2,'current',t)]for t in range(7))),'H_In_h2_prior_correct_both':int(all((p[get(b,'one_step',2),3+h]>=.5)==bool(y[get(b,'current',2),3+h])for b in branches)),'H_In_h2_post_correct_both':int(all((p[get(b,'current',2),3+h]>=.5)==bool(y[get(b,'current',2),3+h])for b in branches))});rec['primary_joint3']=int(rec['valid_pair']and rec['joint3_actual']);rec['primary_joint4']=int(rec['valid_pair']and rec['joint4_actual'])
   for b in branches:
    seq=[('current',0),('current',1),('current',2)]+[('suffix',t)for t in [3,4,5,6]];bad=next(((m,t,get(b,m,t))for m,t in seq if not ok[get(b,m,t)]),None)
    if bad:
     m,t,j=bad;prior=get(b,'one_step',t)if t else j;first.append({**meta,'branch':b,'mode':m,'step':t,'truth_relations':y[j].tolist(),'predicted_relation_probabilities':p[j].tolist(),'prior_relation_probabilities':p[prior].tolist(),'current_relation_probabilities':p[get(b,'current',t)].tolist(),'binding':z['binding'][prior].tolist(),'visibility':z['availability'][j].tolist()})
  rr.append(rec)
  for b in branches:
   for t in range(1,length+1):
    ip=get(b,'one_step',t);ic=get(b,'current',t);prev=get(b,'current',t-1);skill=str(z['skill'][ip]);action_object=next((j for j,c in enumerate(COLORS)if skill=='lift_'+c+'_block_table'),None)
    # Place target is unavailable as public input: GT held identity below is used ONLY to define evaluation strata.
    if skill=='place_in_drawer':
     held=np.where(z['cat'][prev,:3]==6)[0];action_object=int(held[0])if len(held)==1 else -2
    for j,c in enumerate(COLORS):
     if action_object==-2 or j==action_object:continue
     stable=bool(y[prev,3+j]and y[ic,3+j]);new=bool(not y[prev,3+j]and y[ic,3+j]);pc=bool(p[ip,3+j]>=.5);post=bool(p[ic,3+j]>=.5)
     ff.append({**meta,'branch':b,'step':t,'object':c,'skill':skill,'stable_trueIn':stable,'new_trueIn':new,'prior_correct':pc,'forgot':bool(stable and pc and not post),'prior':float(p[ip,3+j]),'posterior':float(p[ic,3+j]),'nontarget_moved':bool(z['event_target'][ip,1+j]),'event_valid':bool(z['events_valid'][ip,j]),'predicted_moved':bool(z['event_prob'][ip,1+j]>=.5),'location_category_changed':bool(z['cat'][prev,j]!=z['cat'][ic,j])})
  for j in ids:
   if z['mode'][j]=='current':continue
   pr=np.clip(z['event_prob'][j],1e-9,1-1e-9);yt=z['event_target'][j];brier=(pr-yt)**2;nll=-(yt*np.log(pr)+(1-yt)*np.log1p(-pr));valid=z['events_valid'][j];events.append({**meta,'branch':int(z['branch'][j]),'step':int(z['step'][j]),'mode':str(z['mode'][j]),'status':str(z['status'][j]),'success_brier':float(brier[0]),'event_brier':float(brier[1:][valid].mean())if valid.any()else 0.,'success_nll':float(nll[0]),'event_nll':float(nll[1:][valid].mean())if valid.any()else 0.})
 return rr,ff,events,first
def main():
 guard();verify();roles=[];forget=[];events=[];first=[]
 for p in sorted((S/'results').glob('predictions_*.npz')):
  a,b,c,d=analyze(p);roles+=a;forget+=b;events+=c;first+=d
 csvwrite(S/'results/root_metrics.csv',roles);csvwrite(S/'results/retention_opportunities.csv',forget);csvwrite(S/'results/outcome_metrics.csv',events);save(S/'results/first_errors.json',first);families=[];summaries=[]
 for split in ['train','development','historical_evaluation']:
  for kind in ['core','role']:
   for seed in [0,1]:
    for arm in ARMS:
     rr=[r for r in roles if r['split']==split and r['kind']==kind and r['seed']==seed and r['arm']==arm];meta={'split':split,'kind':kind,'seed':seed,'arm':arm};metrics=[k for k in rr[0]if k not in ['split','kind','seed','arm','root_id','family_id','valid_pair']];fam=[]
     for fid in sorted({r['family_id']for r in rr}):
      xs=[r for r in rr if r['family_id']==fid];f={**meta,'family_id':fid,**{k:float(np.mean([r[k]for r in xs]))for k in metrics}};fam.append(f);families.append(f)
     fs=[r for r in forget if all(r[k]==v for k,v in meta.items())];stable=[r for r in fs if r['stable_trueIn']];pc=[r for r in stable if r['prior_correct']];mov=[r for r in fs if r['event_valid']and r['nontarget_moved']];neg=[r for r in fs if r['event_valid']and not r['nontarget_moved']];summary={**meta,'roots':len(rr),'families':len(fam),**{k:float(np.mean([f[k]for f in fam]))for k in metrics},'stable_In_opportunities':len(stable),'prior_correct_stable_In':len(pc),'forget_count':sum(r['forgot']for r in stable),'forget_fixed_denominator_rate':sum(r['forgot']for r in stable)/max(len(stable),1),'forget_conditional_rate':sum(r['forgot']for r in stable)/max(len(pc),1),'new_trueIn_opportunities':sum(r['new_trueIn']for r in fs),'moved_positives':len(mov),'moved_recall':np.mean([r['predicted_moved']for r in mov])if mov else None,'moved_FPR':np.mean([r['predicted_moved']for r in neg])if neg else None};summaries.append(summary)
 csvwrite(S/'results/family_metrics.csv',families);csvwrite(S/'results/summary.csv',summaries);contrasts=[]
 for metric in ['primary_joint3','joint3_actual','h0_full','h0_relation','normal_whole','H_In_h2_post_correct_both','terminal']:
  diffs=[]
  for seed in [0,1]:
   xs={arm:{r['family_id']:r[metric]for r in families if r['split']=='historical_evaluation'and r['kind']=='role'and r['seed']==seed and r['arm']==arm}for arm in ARMS};assert xs['global'].keys()==xs['object'].keys();d=np.array([xs['object'][fid]-xs['global'][fid]for fid in sorted(xs['global'])]);diffs.append(d);contrasts.append({'metric':metric,'seed':str(seed),**bootstrap(d)})
  contrasts.append({'metric':metric,'seed':'mean',**bootstrap(np.mean(diffs,axis=0))})
 csvwrite(S/'results/contrasts.csv',contrasts);fitchecks=[]
 for r in summaries:
  if r['split']!='train':continue
  for k in ['joint3_actual','h0_full','normal_whole']if r['kind']=='role'else ['h0_full','normal_whole']:fitchecks.append({'kind':r['kind'],'arm':r['arm'],'seed':r['seed'],'metric':k,'value':r[k],'threshold':.9,'passed':r[k]>=.9})
 c=[x for x in contrasts if x['metric']=='primary_joint3'];improved=all(x['difference']>0 for x in c if x['seed']!='mean')and next(x for x in c if x['seed']=='mean')['lo']>0;regressions=[]
 for seed in [0,1]:
  a={r['arm']:r for r in summaries if r['split']=='historical_evaluation'and r['kind']=='role'and r['seed']==seed}
  for k in ['h0_full','normal_whole','H_In_h2_post_correct_both']:
   if a['object'][k]<a['global'][k]:regressions.append({'seed':seed,'metric':k})
  if a['object']['forget_fixed_denominator_rate']>a['global']['forget_fixed_denominator_rate']:regressions.append({'seed':seed,'metric':'forget_fixed_denominator_rate'})
 save(S/'results/analysis.json',{'utc':now(),'training_fit_gate':all(x['passed']for x in fitchecks),'fit_checks':fitchecks,'historical_improvement_gate':improved,'regressions':regressions,'continuation_gate':all(x['passed']for x in fitchecks)and improved and not regressions,'summaries':summaries,'contrasts':contrasts,'historical_exploratory_only':True,'no_new_CD':True});print('ANALYSIS_COMPLETE',flush=True)
if __name__=='__main__':main()
