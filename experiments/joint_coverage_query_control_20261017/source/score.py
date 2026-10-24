"""Fixed all-fact and root-paired readout analysis; no model or point selection."""
from config import *
import csv,collections,itertools
GROUPS={'full9':list(range(9)),'Holding':list(range(3)),'In':list(range(3,6)),'drawer':list(range(6,9))}
def rate(n,N):return n/N if N else None
def csvout(name,rows):
 if rows:
  with (R/(name+'.csv')).open('w')as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def output(name,rows):save(E/'results'/(name+'.json'),rows);csvout(name,rows)
def confusion(y,p,v):
 y,p,v=map(lambda a:np.asarray(a,bool),(y,p,v));tp=int((v&y&p).sum());fn=int((v&y&~p).sum());fp=int((v&~y&p).sum());tn=int((v&~y&~p).sum());return dict(TP=tp,FN=fn,FP=fp,TN=tn,positive_N=tp+fn,negative_N=fp+tn,positive_recall=rate(tp,tp+fn),negative_recall=rate(tn,fp+tn))
def measure(meta,y,p,v,roots):
 y,p,v=map(lambda a:np.asarray(a,bool),(y,p,v));roots=np.asarray(roots);row=meta|{'states':len(y),'roots':len(set(roots))};facts=[]
 for group,ix in GROUPS.items():
  eligible=v[:,ix].all(1);ok=eligible&(p[:,ix]==y[:,ix]).all(1);rates=[float(ok[roots==r].sum()/eligible[roots==r].sum())for r in sorted(set(roots))if eligible[roots==r].any()];row.update({group+'_correct':int(ok.sum()),group+'_N':int(eligible.sum()),group+'_root_macro':float(np.mean(rates))if rates else None})
 for j,f in enumerate(FACTS):facts.append(meta|{'fact':f,**confusion(y[:,j],p[:,j],v[:,j])})
 return row,facts
def main():
 verify();assert not(E/'results/actual_metrics.json').exists();actual=read(E/'results/actual_predictions.json');traj=read(E/'results/trajectory_predictions.json');cfg=read(E/'configs/protocol.json');datasets={f.stem:dict(np.load(f))for f in(E/'data/evaluation').glob('*.npz')};metrics=[];facts=[];paired=[];grouped=collections.defaultdict(list);ap={}
 for r in actual:grouped[(r['cohort'],r['seed'],r['query_arm'])].append(r);ap[(r['cohort'],r['seed'],r['query_arm'],r['state_index'])]=r
 for (cohort,seed,arm),rows in sorted(grouped.items()):
  rows=sorted(rows,key=lambda r:r['state_index']);d=datasets[cohort];assert [r['state_index']for r in rows]==list(range(len(d['y'])));y=d['y'].astype(bool);v=d['v'].astype(bool);p=discrete([r['probability']for r in rows]);dual=np.array([r['dual_relation']for r in rows]);held=(y[:,:3]&v[:,:3]).any(1);inside=(y[:,3:6]&v[:,3:6]).any(1);knownH=v[:,:3].all(1);knownIn=v[:,3:6].all(1)
  subsets={'all':np.ones(len(y),bool),'dual':dual,'nondual':~dual,'holding_without_In':held&knownIn&~inside,'empty_hand':knownH&~held}
  for subset,mask in subsets.items():
   if not mask.any():continue
   meta={'cohort':cohort,'seed':seed,'query_arm':arm,'subset':subset};m,ff=measure(meta,y[mask],p[mask],v[mask],d['root_ids'][mask]);metrics.append(m);facts+=ff
   if arm!='coverage':continue
   for base in ['original','control']:
    b=discrete([ap[(cohort,seed,base,i)]['probability']for i in range(len(y))]);eligible=mask&v.all(1);aa=(b==y).all(1);bb=(p==y).all(1);rr=[]
    for root in sorted(set(d['root_ids'][eligible])):
     ii=eligible&(d['root_ids']==root);rr.append({'root_id':str(root),'N':int(ii.sum()),'base_correct':int((ii&aa).sum()),'coverage_correct':int((ii&bb).sum()),'gains':int((ii&~aa&bb).sum()),'losses':int((ii&aa&~bb).sum()),'difference':float((bb[ii].astype(int)-aa[ii]).mean())})
    paired.append(meta|{'baseline':base,'eligible_states':int(eligible.sum()),'root_N':len(rr),'gains':int((eligible&~aa&bb).sum()),'losses':int((eligible&aa&~bb).sum()),'root_macro_difference':float(np.mean([r['difference']for r in rr]))if rr else None,'roots_with_net_gain':sum(r['difference']>0 for r in rr),'roots_with_net_loss':sum(r['difference']<0 for r in rr),'per_root':rr})
 output('actual_metrics',metrics);output('actual_fact_confusions',facts);output('actual_paired',paired)
 # Fixed saved world-model outputs: full nine-fact correctness at every skill boundary.
 labs=read(COMBO/'data/combo/_private/labels.json');stage=[];tfacts=[];terminal=[];rootrows=[];tg=collections.defaultdict(list)
 for r in traj:tg[(r['split'],r['seed'],r['world_model'],r['query_arm'],r['branch'])].append(r)
 for key,rows in sorted(tg.items()):
  split,seed,wm,arm,branch=key;meta=dict(split=split,seed=seed,world_model=wm,query_arm=arm,branch=branch);byroot=collections.defaultdict(list)
  for r in rows:byroot[r['root_id']].append(r)
  assert len(byroot)==6
  for path in PATHS:
   for root,rr in sorted(byroot.items()):
    rr=sorted(rr,key=lambda x:x['slot']);assert [r['slot']for r in rr]==[0,1,2,3];ys=np.array([labs[str(r['after'])]['y']for r in rr],bool);vs=np.array([labs[str(r['after'])]['v']for r in rr],bool);ps=discrete([r[path+'_probability']for r in rr]);assert vs.all();ok=(ps==ys).all(1).tolist();first=next((j+1 for j,b in enumerate(ok)if not b),None)
    rootrows.append(meta|{'path':path,'root_id':root,'step_correct':ok,'prefix_correct':[all(ok[:j])for j in range(1,5)],'all4_correct':all(ok),'first_wrong_step':first,'first_wrong_facts':[FACTS[j]for j in range(9)if ps[first-1,j]!=ys[first-1,j]]if first else[]})
   for slot in range(4):
    rr=sorted([r for r in rows if r['slot']==slot],key=lambda r:r['root_id']);ys=np.array([labs[str(r['after'])]['y']for r in rr],bool);vs=np.array([labs[str(r['after'])]['v']for r in rr],bool);ps=discrete([r[path+'_probability']for r in rr]);m,ff=measure(meta|{'path':path,'step':slot+1},ys,ps,vs,[r['root_id']for r in rr]);stage.append(m);tfacts+=ff
    if slot==3:
     ar=np.arange(len(rr));held=np.array([COLORS.index(r['other_target']if branch=='place_then_other_lift'else r['target'])for r in rr]);orig=np.array([COLORS.index(r['target'])+3 for r in rr]);row=meta|{'path':path,'roots':len(rr)}
     for title,ix in [('target_Holding',held),('original_In',orig)]:
      row.update({title+'_'+k:v for k,v in confusion(ys[ar,ix],ps[ar,ix],vs[ar,ix]).items()})
     row['other_Holding_false_positive_roots']=int((vs[:,:3]&~ys[:,:3]&ps[:,:3]).any(1).sum());terminal.append(row)
 output('trajectory_root_outcomes',rootrows);output('trajectory_stage_metrics',stage);output('trajectory_fact_confusions',tfacts);output('trajectory_terminal',terminal)
 primary=[];tpairs=[];rg=collections.defaultdict(list)
 for r in rootrows:rg[(r['split'],r['seed'],r['world_model'],r['query_arm'],r['branch'],r['path'])].append(r)
 for key,rows in sorted(rg.items()):
  split,seed,wm,arm,branch,path=key;meta=dict(split=split,seed=seed,world_model=wm,query_arm=arm,branch=branch,path=path);primary.append(meta|{'correct':sum(r['all4_correct']for r in rows),'N':len(rows),'first_error_counts':[sum(r['first_wrong_step']==j for r in rows)for j in range(1,5)],'prefix_correct_counts':[sum(r['prefix_correct'][j]for r in rows)for j in range(4)]})
  if arm!='coverage':continue
  for base in ['original','control']:
   lookup={r['root_id']:r for r in rg[(split,seed,wm,base,branch,path)]};d=[{'root_id':r['root_id'],'baseline_correct':lookup[r['root_id']]['all4_correct'],'coverage_correct':r['all4_correct'],'difference':int(r['all4_correct'])-int(lookup[r['root_id']]['all4_correct'])}for r in rows];tpairs.append(meta|{'baseline':base,'N':len(d),'gains':sum(r['difference']==1 for r in d),'losses':sum(r['difference']==-1 for r in d),'difference':float(np.mean([r['difference']for r in d])),'per_root':d})
 output('trajectory_primary',primary);output('trajectory_paired',tpairs)
 # Cross-model actual probabilities must coincide: only frozen predicted z differs.
 lookup={(r['id'],r['seed'],r['query_arm'],r['world_model']):r for r in traj};delta=max(float(np.max(np.abs(np.array(r['actual_probability'])-lookup[(r['id'],r['seed'],r['query_arm'],'latent')]['actual_probability'])))for r in traj if r['world_model']=='mixed');assert delta<1e-6
 save(E/'results/scoring_audit.json',{'utc':now(),'actual_state_rows':len(actual),'trajectory_rows':len(traj),'actual_across_worldmodel_max_delta':delta,'root_clusters_per_split':6,'seeds_branches_and_states_not_independent_roots':True,'unknown_labels_excluded_not_false':True,'all9_facts_scored':True,'D_seen_diagnostic_only':True,'new_gate':False,'adoption':False,'no_bootstrap_precision_claim_with_six_roots':True});print('QUERY_CONTROL_SCORED',len(metrics),len(facts),len(primary),len(tfacts),flush=True)
if __name__=='__main__':main()
