from common import *
import collections
def stats(rr,values):
 root=collections.defaultdict(list)
 for r,v in zip(rr,values):root[r['root_id']].append(float(v))
 v=np.array([np.mean(a)for a in root.values()]);rng=np.random.default_rng(2610192200);ci=np.quantile(v[rng.integers(len(v),size=(2000,len(v)))].mean(1),[.025,.975])if len(v)>1 else[np.nan,np.nan]
 return {'n':len(rr),'roots':len(root),'count':sum(values),'root_mean':float(v.mean()),'ci_low':float(ci[0]),'ci_high':float(ci[1])}
def main():
 sys.addaudithook(guard);a=dict(np.load(S/'data/paired_predictions.npz'));datasets={s:dict(np.load(G/f'data/dataset_{s}.npz'))for s in SPLITS};dix={(str(e),t):d['y'][i,t]for d in datasets.values()for i,e in enumerate(d['episode_id'])for t in range(5)};vis={(r['episode_id'],r['boundary'],r['color']):r for r in read(S/'data/visibility.json')};records=[]
 for i in range(len(a['prob'])):
  r={k:plain(a[k][i])for k in a};r['scene']=r['root_id'][0];p=rel_from_prob(r['prob']);y=rel_from_cat(np.asarray(r['cat'])[:4]);r['task_prob']=p.tolist();r['truth']=y.tolist();r['correct']=correct(p,y).tolist();z=hard(p);r['extra_held']=int(((z[:3]>0)&(y[:3]==0)).sum());r['extra_in']=int(((z[3:6]>0)&(y[3:6]==0)).sum());r['missing_held']=int(((z[:3]==0)&(y[:3]>0)).sum());r['missing_in']=int(((z[3:6]==0)&(y[3:6]>0)).sum());r['held_positive']=bool(y[:3].any());r['drawer_positive']=bool(y[3:6].any());r['stable_held']=bool(r['target']>0 and y[:3].any()and np.array_equal(y[:3],dix[r['episode_id'],r['target']-1][:3]));r['stable_in']=bool(r['target']>0 and y[3:6].any()and np.array_equal(y[3:6],dix[r['episode_id'],r['target']-1][3:6]));r['occluded_true_relation']=any((y[j]or y[3+j])and not vis[r['episode_id'],r['target'],COLORS[j]]['visible_in_window']for j in range(3));records.append(r)
 save(S/'data/records.json',records);lookup={(r['seed'],r['episode_id'],r['condition'],r['mode'],r['origin'],r['target']):r for r in records};current=[r for r in records if r['mode']=='current'and r['target']>0];summary=[];paired=[]
 def cohort(r,name):return name=='all'or name=='after_failure'and r['target_previous_status']=='failed'or name=='after_first_failure'and r['after_first_failure']or name=='after_refusal'and r['target_previous_status']=='refused'or name=='after_success'and r['target_previous_status']=='success'or name=='held_positive'and r['held_positive']or name=='drawer_positive'and r['drawer_positive']or name=='stable_held'and r['stable_held']or name=='stable_in'and r['stable_in']or name=='occluded_true_relation'and r['occluded_true_relation']
 cohorts=['all','after_failure','after_first_failure','after_refusal','after_success','held_positive','drawer_positive','stable_held','stable_in','occluded_true_relation']
 for seed in [0,1]:
  for scene in ['all','A','B']:
   for split in ['all']+SPLITS:
    for name in cohorts:
     base=[r for r in current if r['seed']==seed and r['condition']=='continuous'and(scene=='all'or r['scene']==scene)and(split=='all'or r['split']==split)and cohort(r,name)]
     if not base:continue
     for field,j in zip(FIELDS,range(4)):
      cc=[];zz=[];pp=[]
      for r in base:
       reset=lookup[seed,r['episode_id'],'reset','current',r['origin'],r['target']];prior=lookup[seed,r['episode_id'],'prior','prior_current',r['origin'],r['target']];cc.append(r['correct'][j]);zz.append(reset['correct'][j]);pp.append(prior['correct'][j])
      key={'seed':seed,'scene':scene,'split':split,'cohort':name,'field':field}
      for cond,vals in [('prior',pp),('continuous',cc),('reset',zz)]:summary.append({**key,'condition':cond,**stats(base,vals)})
      dif=np.array(zz,dtype=int)-np.array(cc,dtype=int);paired.append({**key,'continuous_correct':sum(cc),'reset_correct':sum(zz),'corrected':int((dif==1).sum()),'continuous_wrong':len(cc)-sum(cc),'regressed':int((dif==-1).sum()),'continuous_right':sum(cc),**stats(base,dif.tolist())})
 csvwrite(S/'results/current_metrics.csv',summary);csvwrite(S/'results/current_pairs.csv',paired)
 future=[]
 for seed in [0,1]:
  for cond in ['continuous','reset']:
   for mode in ['one_step','free_root','free_suffix']:
    for horizon in range(1,5):
     for origin_status in ['all','failed','refused','success']:
      rr=[r for r in records if(r['seed'],r['condition'],r['mode'],r['horizon'])==(seed,cond,mode,horizon)and(origin_status=='all'or r['origin_previous_status']==origin_status)]
      if not rr:continue
      for field,j in zip(FIELDS,range(4)):future.append({'seed':seed,'condition':cond,'mode':mode,'horizon':horizon,'origin_previous_status':origin_status,'field':field,**stats(rr,[r['correct'][j]for r in rr])})
 csvwrite(S/'results/forecast_metrics.csv',future)
 extra=[]
 for seed in [0,1]:
  for condition in ['continuous','reset']:
   for name in cohorts:
    rr=[r for r in current if r['seed']==seed and r['condition']==condition and cohort(r,name)]
    if rr:
     for field in ['extra_held','extra_in','missing_held','missing_in']:extra.append({'seed':seed,'condition':condition,'cohort':name,'field':field,**stats(rr,[r[field]for r in rr])})
 csvwrite(S/'results/extra_relations.csv',extra)
 memory=[]
 for seed in [0,1]:
  for field,offset in [('holding',0),('InDrawer',3)]:
   for visibility in ['all','visible','occluded']:
    for stable in [False,True]:
     rr=[];cp=[];rp=[]
     for r in current:
      if r['seed']!=seed or r['condition']!='continuous':continue
      reset=lookup[seed,r['episode_id'],'reset','current',r['origin'],r['target']]
      for j,color in enumerate(COLORS):
       vv=vis[r['episode_id'],r['target'],color]['visible_in_window'];is_stable=bool(r['truth'][offset+j]and dix[r['episode_id'],r['target']-1][offset+j])
       if not r['truth'][offset+j]or(visibility!='all'and vv!=(visibility=='visible'))or(stable and not is_stable):continue
       rr.append({**r,'color':color});cp.append(r['task_prob'][offset+j]>=.5);rp.append(reset['task_prob'][offset+j]>=.5)
     if rr:memory.append({'seed':seed,'field':field,'visibility':visibility,'stable_only':stable,'continuous_TP':sum(cp),'reset_TP':sum(rp),'lost_positive':sum(a and not b for a,b in zip(cp,rp)),'restored_positive':sum(not a and b for a,b in zip(cp,rp)),**stats(rr,[int(b)-int(a)for a,b in zip(cp,rp)])})
 csvwrite(S/'results/memory_retention.csv',memory)
 failure=[r for r in records if r['mode']in ['current','prior_current']and r['target']>0 and r['target_previous_status']=='failed'];save(S/'results/failure_boundaries.json',failure)
 # Cases chosen by first effect, then independent roots; do not omit regressions.
 cases=[];used=set();candidates=[r for r in current if r['condition']=='continuous'];specs=[('failure_observation_correction',lambda r:r['target_previous_status']=='failed'and not r['correct'][3]and lookup[r['seed'],r['episode_id'],'reset','current',r['origin'],r['target']]['correct'][3]),('lost_drawer_memory',lambda r:r['drawer_positive']and r['correct'][1]and not lookup[r['seed'],r['episode_id'],'reset','current',r['origin'],r['target']]['correct'][1]),('lost_holding_memory',lambda r:r['held_positive']and r['correct'][0]and not lookup[r['seed'],r['episode_id'],'reset','current',r['origin'],r['target']]['correct'][0]),('refusal_regression',lambda r:r['target_previous_status']=='refused'and r['correct'][3]and not lookup[r['seed'],r['episode_id'],'reset','current',r['origin'],r['target']]['correct'][3])]
 for name,pred in specs:
  pool=[r for r in candidates if r['root_id']not in used and pred(r)]
  if pool:
   r=min(pool,key=lambda r:(r['seed'],r['episode_id'],r['target']));c={'case_type':name,'seed':r['seed'],'split':r['split'],'root_id':r['root_id'],'episode_id':r['episode_id'],'first_selected_boundary':r['target'],'rows':[a for a in records if(a['seed'],a['episode_id'])==(r['seed'],r['episode_id'])]};cases.append(c);used.add(r['root_id'])
 if not any(c['episode_id']=='B_coverage_gap_04_b0'for c in cases):cases.append({'case_type':'target_failure_no_full_correction','seed':0,'split':'gap','root_id':'B_coverage_gap_04','episode_id':'B_coverage_gap_04_b0','first_selected_boundary':1,'rows':[r for r in records if(r['seed'],r['episode_id'])==(0,'B_coverage_gap_04_b0')]})
 save(S/'results/cases.json',cases);save(S/'results/analysis_complete.json',{'utc':now(),'rows':len(records),'current_metric_rows':len(summary),'cases':len(cases),'new_forwards':0});print('ANALYSIS_COMPLETE',len(records),[(c['case_type'],c['episode_id'],c['first_selected_boundary'])for c in cases],flush=True)
if __name__=='__main__':main()
