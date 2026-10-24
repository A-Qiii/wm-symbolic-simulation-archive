from config import *
import collections,csv
GROUPS={'full9':list(range(9)),'Holding':list(range(3)),'In':list(range(3,6)),'drawer':list(range(6,9))}
def conf(y,p,v):
 y,p,v=map(lambda x:np.asarray(x,bool),(y,p,v));tp=int((v&y&p).sum());fn=int((v&y&~p).sum());fp=int((v&~y&p).sum());tn=int((v&~y&~p).sum());return {'TP':tp,'FN':fn,'FP':fp,'TN':tn,'positive_recall':tp/(tp+fn)if tp+fn else None,'FPR':fp/(fp+tn)if fp+tn else None,'unknown':int((~v).sum())}
def output(name,rows):
 save(E/'results'/f'{name}.json',rows)
 if rows:
  with(R/f'{name}.csv').open('w')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def metric(meta,y,v,p,roots):
 row=meta|{'states':len(y),'roots':len(set(roots))};roots=np.array(roots)
 for group,ix in GROUPS.items():
  eligible=v[:,ix].all(1);ok=eligible&(p[:,ix]==y[:,ix]).all(1);rates=[float(ok[roots==r].sum()/eligible[roots==r].sum())for r in sorted(set(roots))if eligible[roots==r].any()];row.update({group+'_correct':int(ok.sum()),group+'_N':int(eligible.sum()),group+'_root_macro':float(np.mean(rates))if rates else None})
 return row
def main():
 verify();assert not(E/'results/actual_metrics.json').exists();actual=read(E/'results/actual_predictions.json');traj=read(E/'results/trajectory_predictions.json');datasets={f.stem:dict(np.load(f))for f in(WHOLE/'data/evaluation').glob('*.npz')};groups=collections.defaultdict(list);metrics=[];facts=[];heldmetrics=[];pairs=[];fixedmax=0
 for r in actual:groups[(r['cohort'],r['seed'],r['query_arm'])].append(r)
 for(cohort,seed,arm),rows in sorted(groups.items()):
  rows=sorted(rows,key=lambda r:r['state_index']);d=datasets[cohort];y=d['y'].astype(bool);v=d['v'].astype(bool);pp=np.array([r['probability']for r in rows]);p=discrete(pp);roots=d['root_ids'];dual=np.array([r['dual_relation']for r in rows]);orig=sorted(groups[(cohort,seed,'original')],key=lambda r:r['state_index']);op=np.array([r['probability']for r in orig]);original=discrete(op);maskhd=[0,1,2,6,7,8];ceiling=v.all(1)&(original[:,maskhd]==y[:,maskhd]).all(1)
  if arm.startswith('in_'):assert np.array_equal(pp[:,maskhd],op[:,maskhd])
  for subset,mask in [('all',np.ones(len(y),bool)),('dual',dual),('nondual',~dual)]:
   if not mask.any():continue
   meta=dict(cohort=cohort,seed=seed,query_arm=arm,subset=subset);m=metric(meta,y[mask],v[mask],p[mask],roots[mask]);m['original_Holding_drawer_ceiling_states']=int((mask&ceiling).sum());metrics.append(m)
   for j,f in enumerate(FACTS):facts.append(meta|{'fact':f,**conf(y[mask,j],p[mask,j],v[mask,j])})
   if arm!='in_coverage':continue
   for base in ['in_control','original','whole_coverage']:
    b=discrete([r['probability']for r in sorted(groups[(cohort,seed,base)],key=lambda r:r['state_index'])])
    for group,ix in [('full9',list(range(9))),('In',list(range(3,6)))]:
     eligible=mask&v[:,ix].all(1);aa=(b[:,ix]==y[:,ix]).all(1);bb=(p[:,ix]==y[:,ix]).all(1);rr=[]
     for root in sorted(set(roots[eligible])):
      take=eligible&(roots==root);rr.append({'root_id':str(root),'N':int(take.sum()),'base_correct':int((take&aa).sum()),'candidate_correct':int((take&bb).sum()),'gains':int((take&~aa&bb).sum()),'losses':int((take&aa&~bb).sum()),'difference':float((bb[take].astype(int)-aa[take]).mean())})
     pairs.append(meta|{'baseline':base,'metric':group,'N':int(eligible.sum()),'root_N':len(rr),'gains':int((eligible&~aa&bb).sum()),'losses':int((eligible&aa&~bb).sum()),'root_macro_difference':float(np.mean([r['difference']for r in rr]))if rr else None,'roots_net_gain':sum(r['difference']>0 for r in rr),'roots_net_loss':sum(r['difference']<0 for r in rr),'per_root':rr})
  held=y[:,:3]&v[:,:3];hknown=v[:,:3];anyheld=held.any(1,keepdims=True);conditions={'all':np.ones_like(held),'self_held':held,'not_self_held':hknown&~held,'other_held':hknown&~held&anyheld,'empty_hand':np.repeat((hknown.all(1)&~held.any(1))[:,None],3,1),'Holding_unknown':~hknown}
  for condition,take in conditions.items():
   for color,ci in [('all',None),*[(c,i)for i,c in enumerate(COLORS)]]:
    mask=take.copy()
    if ci is not None:mask[:,[i for i in range(3)if i!=ci]]=False
    iy=y[:,3:6][mask];iv=v[:,3:6][mask];ip=p[:,3:6][mask];rootcells=np.repeat(roots[:,None],3,axis=1)[mask];negative=iv&~iy;rootFPR=[float(ip[(rootcells==r)&negative].mean())for r in sorted(set(rootcells))if((rootcells==r)&negative).any()]
    heldmetrics.append({'cohort':cohort,'seed':seed,'query_arm':arm,'condition':condition,'color':color,'candidate_object_cells':int(mask.sum()),'valid_roots':len(set(rootcells[iv])),'root_macro_FPR':float(np.mean(rootFPR))if rootFPR else None,**conf(iy,ip,iv)})
 output('actual_metrics',metrics);output('actual_fact_confusions',facts);output('actual_paired',pairs);output('In_condition_confusions',heldmetrics)
 labs=read(COMBO/'data/combo/_private/labels.json');tg=collections.defaultdict(list);rootsout=[];stages=[];tfacts=[];tself=[]
 for r in traj:tg[(r['split'],r['seed'],r['world_model'],r['query_arm'],r['branch'])].append(r)
 origlookup={(r['seed'],r['world_model'],r['id']):r for r in traj if r['query_arm']=='original'}
 for key,rows in sorted(tg.items()):
  split,seed,wm,arm,branch=key;meta=dict(split=split,seed=seed,world_model=wm,query_arm=arm,branch=branch);byroot=collections.defaultdict(list)
  for r in rows:byroot[r['root_id']].append(r)
  for path in PATHS:
   if arm.startswith('in_'):
    for r in rows:assert all(r[path+'_probability'][j]==origlookup[(seed,wm,r['id'])][path+'_probability'][j]for j in maskhd)
   for root,rr in sorted(byroot.items()):
    rr=sorted(rr,key=lambda r:r['slot']);assert [r['slot']for r in rr]==[0,1,2,3];y=np.array([labs[str(r['after'])]['y']for r in rr],bool);v=np.array([labs[str(r['after'])]['v']for r in rr],bool);assert v.all();p=discrete([r[path+'_probability']for r in rr]);op=discrete([origlookup[(seed,wm,r['id'])][path+'_probability']for r in rr]);ok=(p==y).all(1).tolist();iok=(p[:,3:6]==y[:,3:6]).all(1).tolist();first=next((i+1 for i,b in enumerate(ok)if not b),None);rootsout.append(meta|{'path':path,'root_id':root,'step_correct':ok,'step_In3_correct':iok,'all4_correct':all(ok),'all4_In3_correct':all(iok),'original_Holding_drawer_ceiling':bool((op[:,maskhd]==y[:,maskhd]).all()),'first_wrong_step':first,'first_wrong_facts':[FACTS[j]for j in range(9)if p[first-1,j]!=y[first-1,j]]if first else[]})
   for slot in range(4):
    rr=sorted([r for r in rows if r['slot']==slot],key=lambda r:r['root_id']);y=np.array([labs[str(r['after'])]['y']for r in rr],bool);v=np.array([labs[str(r['after'])]['v']for r in rr],bool);p=discrete([r[path+'_probability']for r in rr]);sm=meta|{'path':path,'step':slot+1};stages.append(metric(sm,y,v,p,[r['root_id']for r in rr]))
    for j,f in enumerate(FACTS):tfacts.append(sm|{'fact':f,**conf(y[:,j],p[:,j],v[:,j])})
    held=y[:,:3]&v[:,:3];tself.append(sm|{'condition':'self_held','object_cells':int(held.sum()),**conf(y[:,3:6][held],p[:,3:6][held],v[:,3:6][held])})
 output('trajectory_root_outcomes',rootsout);output('trajectory_stage_metrics',stages);output('trajectory_fact_confusions',tfacts);output('trajectory_self_held_In',tself);rg=collections.defaultdict(list);primary=[];tp=[]
 for r in rootsout:rg[(r['split'],r['seed'],r['world_model'],r['query_arm'],r['branch'],r['path'])].append(r)
 for key,rr in sorted(rg.items()):
  split,seed,wm,arm,branch,path=key;meta=dict(split=split,seed=seed,world_model=wm,query_arm=arm,branch=branch,path=path);primary.append(meta|{'N':len(rr),'full9_all4_correct':sum(r['all4_correct']for r in rr),'In3_all4_correct':sum(r['all4_In3_correct']for r in rr),'original_Holding_drawer_ceiling':sum(r['original_Holding_drawer_ceiling']for r in rr)})
  if arm!='in_coverage':continue
  for base in ['in_control','original','whole_coverage']:
   lookup={r['root_id']:r for r in rg[(split,seed,wm,base,branch,path)]}
   for metricname,field in [('full9','all4_correct'),('In','all4_In3_correct')]:
    d=[int(r[field])-int(lookup[r['root_id']][field])for r in rr];tp.append(meta|{'baseline':base,'metric':metricname,'N':len(d),'gains':d.count(1),'losses':d.count(-1),'net':sum(d),'root_ids':[r['root_id']for r in rr],'root_differences':d})
 output('trajectory_primary',primary);output('trajectory_paired',tp);save(E/'results/scoring_audit.json',{'utc':now(),'Holding_drawer_bitwise_unchanged_all_actual_and_predicted_inputs':True,'trueHolding_used_only_for_reporting_subgroups':True,'invalid_In_cells_excluded':True,'all9_and_In3_scored':True,'no_adoption':True,'D_seen_only':True});print('OBJECT_IN_SCORED',len(metrics),len(heldmetrics),len(primary),len(tfacts),flush=True)
if __name__=='__main__':main()
