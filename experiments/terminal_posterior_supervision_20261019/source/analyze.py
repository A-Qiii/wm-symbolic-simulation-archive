from common import *
def rootstats(rr,v):
 groups=collections.defaultdict(list)
 for r,x in zip(rr,v):groups[r['root_id']].append(float(x))
 a=np.array([np.mean(x)for x in groups.values()]);rng=np.random.default_rng(2610192200);ci=np.quantile(a[rng.integers(len(a),size=(2000,len(a)))].mean(1),[.025,.975])if len(a)>1 else [np.nan,np.nan]
 return {'n':len(rr),'roots':len(groups),'count':float(np.sum(v)),'root_mean':float(a.mean()),'ci_low':float(ci[0]),'ci_high':float(ci[1])}
def main():
 guard('eval');episodes={}
 for split in ['train','validation']:
  for e in read(S/f'data/episodes_{split}.json'):episodes[e['episode_id']]=e
 by=collections.defaultdict(list)
 for r in read(R/'data/data_manifest.json')['records']:
  if r['split']in ['old_band','gap']:by[r['episode_id']].append(r)
 for eid,rr in sorted(by.items()):
  rr.sort(key=lambda r:r['step']);episodes[eid]={'episode_id':eid,'root_id':rr[0]['root_id'],'rows':rr,'labels':[read(r['labels_path'])for r in rr]}
 records=[]
 for path in sorted((S/'data').glob('predictions_*_seed*.npz')):
  match=re.match(r'predictions_(.+)_seed([01])_(train|validation|old_band|gap)\.npz',path.name)
  if not match:continue
  variant,seed,split=match.groups();a=dict(np.load(path))
  for i,mode in enumerate(a['mode']):
   t=int(a['step'][i])
   if mode not in ['current','one_step','free1','free2','free3','free4']:continue
   origin=t if mode=='current'else t-1 if mode=='one_step'else 0;mode='free_root'if str(mode).startswith('free')else str(mode)
   records.append({'seed':int(seed),'variant':variant,'split':split,'root_id':str(a['root_id'][i]),'episode_id':str(a['episode_id'][i]),'mode':mode,'origin':origin,'target':t,'horizon':t-origin,'prob':a['prob'][i],'cat':a['cat'][i]})
 a=dict(np.load(S/'data/extra_predictions.npz'))
 for i in range(len(a['prob'])):records.append({k:plain(a[k][i])for k in a})
 for seed in [0,1]:
  for arm in ARMS:
   if not any(r['seed']==seed and r['variant']==arm+'_final750'for r in records):records.extend([{**r,'variant':arm+'_final750'}for r in records if r['seed']==seed and r['variant']==arm])
 for r in records:
  e=episodes[r['episode_id']];t=r['target'];origin=r['origin'];ls=e['labels'];current=ls[0]['before']if t==0 else ls[t-1]['after'];before=ls[0]['before']if t<=1 else ls[t-2]['after'];assert np.array_equal(r['cat'],current['categorical'])
  def status(l):return 'success'if l['success']else'failed'if l['control_samples_including_start']>1 else'refused'
  r['target_status']='root'if not t else status(ls[t-1]);r['origin_status']='root'if not origin else status(ls[origin-1]);r['after_failure']=any(status(l)=='failed'for l in ls[:t]);r['task_prob']=rel(r['prob']);r['truth']=truth(r['cat']);r['before_truth']=truth(before['categorical']);r['correct']=correct(r['task_prob'],r['truth']);r['nll']=float(nll(r['task_prob'],r['truth']))
  z=r['task_prob'][:6]>=.5;y=r['truth'][:6];r['extra_held']=int((z[:3]&~y[:3].astype(bool)).sum());r['extra_in']=int((z[3:6]&~y[3:6].astype(bool)).sum());r['missing_held']=int((~z[:3]&y[:3].astype(bool)).sum());r['missing_in']=int((~z[3:6]&y[3:6].astype(bool)).sum())
 lookup={(r['seed'],r['variant'],r['episode_id'],r['mode'],r['origin'],r['target']):r for r in records}
 cohorts={'all':lambda r:True,'terminal':lambda r:r['target']==4,'nonterminal':lambda r:0<r['target']<4,'failure':lambda r:r['target_status']=='failed','failure_terminal':lambda r:r['target_status']=='failed'and r['target']==4,'failure_nonterminal':lambda r:r['target_status']=='failed'and r['target']<4,'after_first_failure':lambda r:r['after_failure'],'refused':lambda r:r['target_status']=='refused'}
 fields=['held','InDrawer','drawer','all'];metrics=[];pairs=[]
 for seed in [0,1]:
  for split in ['train','validation','exploration','old_band','gap']:
   for mode in ['current','one_step','free_root','free_suffix','prior_current','original_head_current','original_head_one','original_head_suffix']:
    for horizon in ([0]if mode in ['current','prior_current','original_head_current']else [1]if mode in ['one_step','original_head_one']else range(1,5)):
     base=[r for r in records if r['seed']==seed and r['variant']=='native'and r['mode']==mode and r['horizon']==horizon and r['target']>0 and (r['split']==split or split=='exploration'and r['split']in ['old_band','gap'])]
     if not base:continue
     for cohort,predicate in (cohorts.items()if mode in ['current','prior_current','original_head_current']else [('all',lambda r:True),('from_failure',lambda r:r['origin_status']=='failed')]):
      rr=[r for r in base if predicate(r)]
      if not rr:continue
      for variant in ['native','control','terminal','control_final750','terminal_final750']:
       vv=[lookup[seed,variant,r['episode_id'],mode,r['origin'],r['target']]for r in rr]
       key={'seed':seed,'split':split,'variant':variant,'mode':mode,'horizon':horizon,'cohort':cohort}
       for j,f in enumerate(fields):metrics.append({**key,'field':f,**rootstats(rr,[r['correct'][j]for r in vv]),'mean_nll':float(np.mean([r['nll']for r in vv]))})
      if mode=='current':
       for aa,bb in [('control','terminal'),('native','terminal'),('native','control'),('control_final750','terminal_final750')]:
        av=[lookup[seed,aa,r['episode_id'],mode,r['origin'],r['target']]['correct'][3]for r in rr];bv=[lookup[seed,bb,r['episode_id'],mode,r['origin'],r['target']]['correct'][3]for r in rr];diff=np.array(bv,int)-np.array(av,int)
        pairs.append({'seed':seed,'split':split,'cohort':cohort,'reference':aa,'intervention':bb,'reference_correct':sum(av),'intervention_correct':sum(bv),'corrected':int((diff==1).sum()),'regressed':int((diff==-1).sum()),**rootstats(rr,diff)})
 csvwrite(S/'results/metrics.csv',metrics);csvwrite(S/'results/current_pairs.csv',pairs)
 assimilation=[]
 for seed in [0,1]:
  for split in ['train','validation','exploration']:
   for variant in ['native','control','terminal']:
    base=[r for r in records if r['seed']==seed and r['variant']==variant and r['mode']=='current'and r['target']>0 and(r['split']==split or split=='exploration'and r['split']in ['old_band','gap'])]
    for cohort in ['all','terminal','failure_terminal','failure_nonterminal']:
     rr=[r for r in base if cohorts[cohort](r)]
     if not rr:continue
     prior=[lookup[seed,variant,r['episode_id'],'prior_current',r['origin'],r['target']]for r in rr];aa=np.array([r['correct'][3]for r in prior],int);bb=np.array([r['correct'][3]for r in rr],int);diff=bb-aa
     assimilation.append({'seed':seed,'split':split,'variant':variant,'cohort':cohort,'prior_correct':int(aa.sum()),'posterior_correct':int(bb.sum()),'corrected':int((diff==1).sum()),'regressed':int((diff==-1).sum()),**rootstats(rr,diff)})
 csvwrite(S/'results/assimilation.csv',assimilation)

 vis={(r['episode_id'],r['boundary'],j):r['visible_in_window']for r in read(Q/'data/visibility.json')for j,c in enumerate(['red','blue','pink'])if c==r['color']};memory=[];fp=[]
 for seed in [0,1]:
  for split in ['train','validation','exploration']:
   base=[r for r in records if r['seed']==seed and r['variant']=='native'and r['mode']=='current'and r['target']>0 and(r['split']==split or split=='exploration'and r['split']in ['old_band','gap'])]
   for variant in ['native','control','terminal','control_final750','terminal_final750']:
    rr=[lookup[seed,variant,r['episode_id'],'current',r['origin'],r['target']]for r in base]
    for key in ['extra_held','extra_in','missing_held','missing_in']:fp.append({'seed':seed,'split':split,'variant':variant,'field':key,**rootstats(rr,[r[key]for r in rr])})
    for name,offset in [('held',0),('InDrawer',3)]:
     for visibility in (['all','visible','occluded']if split=='exploration'else ['all']):
      pts=[];vals=[]
      for r in rr:
       for j in range(3):
        if not(r['truth'][offset+j]and r['before_truth'][offset+j]):continue
        if visibility!='all'and vis[r['episode_id'],r['target'],j]!=(visibility=='visible'):continue
        pts.append(r);vals.append(r['task_prob'][offset+j]>=.5)
      if pts:memory.append({'seed':seed,'split':split,'variant':variant,'field':name,'visibility':visibility,**rootstats(pts,vals)})
 csvwrite(S/'results/memory.csv',memory);csvwrite(S/'results/false_relations.csv',fp)
 internal=dict(np.load(S/'data/internal_states.npz'));ix={(int(seed),str(variant),str(e),int(t)):i for i,(seed,variant,e,t)in enumerate(zip(internal['seed'],internal['variant'],internal['episode_id'],internal['boundary']))};changes=[]
 for seed in [0,1]:
  for split in SPLITS:
   for variant in ['control','terminal']:
    ii=np.where((internal['seed']==seed)&(internal['variant']==variant)&(internal['split']==split))[0];jj=[ix[seed,'native',str(internal['episode_id'][i]),int(internal['boundary'][i])]for i in ii]
    changes.append({'seed':seed,'split':split,'variant':variant,'boundaries':len(ii),'h_mean_absolute_change':float(abs(internal['h'][ii]-internal['h'][jj]).mean()),'q_mean_total_variation':float((abs(internal['q'][ii]-internal['q'][jj]).sum(-1)/2).mean()),'E_mean_absolute_change':float(abs(internal['E'][ii]-internal['E'][jj]).mean())})
 csvwrite(S/'results/internal_changes.csv',changes)
 # Deterministic case rules select examples after scoring, never checkpoints.
 cases=[{'reason':'known_actual_failed_grasp_transfer','seed':seed,'episode_id':'B_coverage_gap_04_b0'}for seed in [0,1]];used={'B_coverage_gap_04'}
 for tag,sign in [('first_current_improvement',1),('first_current_regression',-1)]:
  pool=[]
  for r in records:
   if r['variant']!='control'or r['mode']!='current'or r['target']==0 or r['root_id']in used or r['split']=='train':continue
   t=lookup[r['seed'],'terminal',r['episode_id'],'current',r['origin'],r['target']]
   if int(t['correct'][3])-int(r['correct'][3])==sign:pool.append(r)
  if pool:
   r=min(pool,key=lambda r:(r['target'],r['seed'],r['episode_id']));cases.append({'reason':tag,'seed':r['seed'],'episode_id':r['episode_id'],'first_effect_boundary':r['target']});used.add(r['root_id'])
 cases.append({'reason':'representative_training_terminal_placement_failure','seed':0,'episode_id':'A_core1200_02_b2'})
 for c in cases:
  c['root_id']=episodes[c['episode_id']]['root_id'];c['rows']=[r for r in records if r['seed']==c['seed']and r['episode_id']==c['episode_id']and r['variant']in ['native','control','terminal']];c['actions']=[r['skill']for r in episodes[c['episode_id']]['rows']];c['statuses']=['success'if l['success']else'failed'if l['control_samples_including_start']>1 else'refused'for l in episodes[c['episode_id']]['labels']]
  c['first_influential_error']={v:next((r['target']for r in sorted(c['rows'],key=lambda r:r['target'])if r['variant']==v and r['mode']=='current'and not r['correct'][3]),None)for v in ['native','control','terminal']}
 save(S/'results/cases.json',cases);save(S/'data/analysis_records.json',records);save(S/'results/analysis_complete.json',{'utc':now(),'rows':len(records),'metric_rows':len(metrics),'cases':len(cases),'independent_case_roots':len({c['root_id']for c in cases}),'no_new_forwards':True});print('ANALYZED',len(records),flush=True)
if __name__=='__main__':main()
