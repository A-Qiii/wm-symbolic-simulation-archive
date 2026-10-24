from common import *
def analyze_file(path):
 z=dict(np.load(path));p=rel(z['prob']);y=truth(z['cat']);ok=correct(p,y)[:,-1]&z['cat_valid'][:,:4].all(1);held=((p[:,:3]>=.5)==y[:,:3]).all(1);roles=[];forget=[];events=[]
 for rid in sorted(set(z['root_id'])):
  ids=np.where(z['root_id']==rid)[0];first=ids[0];index={(int(z['branch'][i]),str(z['mode'][i]),int(z['step'][i])):i for i in ids};get=lambda b,m,t:index[b,m,t]
  both=lambda keys:all(ok[get(b,m,t)]for b in[1,2]for m,t in keys)
  valid=bool(z['valid_physical_pair'][first]);h=COLORS.index(str(z['held_color'][first]));tcol=COLORS.index(str(z['target_color'][first]))
  rec={'root_id':rid,'family_id':str(z['family_id'][first]),'split':str(z['split'][first]),'held':str(z['held_color'][first]),'target':str(z['target_color'][first]),'valid_pair':valid,'in_arm_training':bool(z['in_arm_training'][first]),
   'h0_full':int(both([('current',0)])),'h0_held':int(all(held[get(b,'current',0)]for b in[1,2])),
   'h2_pair':int(both([('current',2)])),'joint3_actual':int(both([('current',2)]+[('suffix',t)for t in[3,4,5]])),
   'joint4_actual':int(both([('current',2)]+[('suffix',t)for t in[3,4,5,6]])),
   'terminal4':int(both([('suffix',6)])),'normal_whole':int(all(ok[get(1,'current',t)]for t in range(7))),
   'disturbed_whole':int(all(ok[get(2,'current',t)]for t in range(7))),
   'H_In_h2_prior_both':int(all(p[get(b,'one_step',2),3+h]>=.5 for b in[1,2])),
   'H_In_h2_post_both':int(all(p[get(b,'current',2),3+h]>=.5 for b in[1,2])),
   'H_In_h2_truth_both':int(all(y[get(b,'current',2),3+h]for b in[1,2])),
   'H_In_h2_prior_correct_both':int(all((p[get(b,'one_step',2),3+h]>=.5)==bool(y[get(b,'current',2),3+h]) for b in[1,2])),
   'H_In_h2_post_correct_both':int(all((p[get(b,'current',2),3+h]>=.5)==bool(y[get(b,'current',2),3+h]) for b in[1,2])),
   'actual_goal_normal':int(y[get(1,'current',6),3+h]and y[get(1,'current',6),3+tcol]and not y[get(1,'current',6),:3].any()and y[get(1,'current',6),6:9].argmax()==0),
   'actual_goal_disturbed':int(y[get(2,'current',6),3+h]and y[get(2,'current',6),3+tcol]and not y[get(2,'current',6),:3].any()and y[get(2,'current',6),6:9].argmax()==0)}
  rec['primary_joint3']=int(valid and rec['joint3_actual']);rec['primary_joint4']=int(valid and rec['joint4_actual']);roles.append(rec)
  for b in[1,2]:
   for t in range(1,7):
    ip=get(b,'one_step',t);ic=get(b,'current',t);prev=get(b,'current',t-1);action_object=h if t==1 else tcol if t<=5 else None
    for j,c in enumerate(COLORS):
     if j!=action_object and y[prev,3+j]and y[ic,3+j]and p[ip,3+j]>=.5 and p[ic,3+j]<.5:forget.append({'root_id':rid,'family_id':rec['family_id'],'split':rec['split'],'branch':b,'step':t,'object':c,'prior':float(p[ip,3+j]),'posterior':float(p[ic,3+j])})
  for i in ids:
   if str(z['mode'][i])not in['one_step','suffix']:continue
   pr=np.clip(z['event_prob'][i],1e-9,1-1e-9);yt=z['event_target'][i];brier=(pr-yt)**2;nll=-(yt*np.log(pr)+(1-yt)*np.log1p(-pr))
   events.append({'family_id':rec['family_id'],'root_id':rid,'split':rec['split'],'branch':int(z['branch'][i]),'mode':str(z['mode'][i]),'step':int(z['step'][i]),'status':str(z['status'][i]),'hidden_fault':bool(z['hidden_fault_action'][i]),'success_brier':float(brier[0]),'event_brier':float(brier[1:].mean()),'success_nll':float(nll[0]),'event_nll':float(nll[1:].mean())})
 return roles,forget,events
def bootstrap(diffs):
 a=np.asarray(diffs,float);rng=np.random.default_rng(261019551);bs=a[rng.integers(len(a),size=(2000,len(a)))].mean(1);return {'difference':float(a.mean()),'lo':float(np.quantile(bs,.025)),'hi':float(np.quantile(bs,.975)),'families':len(a)}
def main():
 guard('analyze');verify_freeze();seal=read(S/'configs/models_frozen.json');allroles=[];allforget=[];allevents=[];families=[]
 for f in sorted((S/'results').glob('predictions_*_seed*_*.npz')):
  match=re.fullmatch(r'predictions_(random|balanced)_step(\d+)_seed(\d+)_(.*)\.npz',f.name)
  if not match:continue
  arm,step,seed,split=match.groups();step=int(step);seed=int(seed);chosen=next(x for x in seal['selection']if x['arm']==arm and x['seed']==seed)['update'];meta={'arm':arm,'update':step,'seed':seed,'selected':step==chosen}
  rr,ff,ee=analyze_file(f);allroles.extend({**meta,**x}for x in rr);allforget.extend({**meta,**x}for x in ff);allevents.extend({**meta,**x}for x in ee)
  groups=[('all',rr)]+([('trained_roles',[x for x in rr if x['in_arm_training']]),('untrained_roles',[x for x in rr if not x['in_arm_training']])]if split=='train'else[])
  keys=[k for k in rr[0]if k not in ['root_id','family_id','split','held','target','valid_pair','in_arm_training']]
  for subset,rows in groups:
   for fid in sorted({r['family_id']for r in rows}):
    r=[x for x in rows if x['family_id']==fid];families.append({**meta,'split':split,'subset':subset,'family_id':fid,'roles':len(r),'valid_pairs':sum(x['valid_pair']for x in r),**{k:float(np.mean([x[k]for x in r]))for k in keys},'all_roles_primary_joint3':int(all(x['primary_joint3']for x in r))})
 csvwrite(S/'results/role_metrics.csv',allroles);csvwrite(S/'results/family_metrics.csv',families);csvwrite(S/'results/forgetting_events.csv',allforget);csvwrite(S/'results/outcome_metrics.csv',allevents)
 contrasts=[]
 for split in ['development','locked_test']:
  for selection in ['selected','fixed1000']:
   rr=[x for x in families if x['split']==split and x['subset']=='all'and(x['selected']if selection=='selected'else x['update']==1000)]
   if not rr:continue
   for metric in ['primary_joint3','joint3_actual','primary_joint4','h0_full','h0_held','normal_whole','H_In_h2_post_correct_both','terminal4']:
    ds={}
    for seed in[0,1]:
     aa={r['family_id']:r[metric]for r in rr if r['arm']=='random'and r['seed']==seed};bb={r['family_id']:r[metric]for r in rr if r['arm']=='balanced'and r['seed']==seed};assert aa.keys()==bb.keys();ds[seed]={fid:bb[fid]-aa[fid]for fid in aa};contrasts.append({'split':split,'selection':selection,'seed':seed,'metric':metric,**bootstrap(list(ds[seed].values()))})
    contrasts.append({'split':split,'selection':selection,'seed':'seed_mean','metric':metric,**bootstrap([(ds[0][fid]+ds[1][fid])/2 for fid in ds[0]])})
 csvwrite(S/'results/contrasts.csv',contrasts);save(S/'results/analysis_complete.json',{'utc':now(),'role_rows':len(allroles),'family_rows':len(families),'forgetting_events':len(allforget),'contrasts':contrasts,'cluster_unit':'geometry_family','no_new_selection':True});print('ANALYSIS_COMPLETE',len(families),flush=True)
if __name__=='__main__':main()
