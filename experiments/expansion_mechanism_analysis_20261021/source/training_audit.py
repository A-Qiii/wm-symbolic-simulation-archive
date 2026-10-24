from pathlib import Path
import json,csv,collections,hashlib
import numpy as np
P=Path('/home/robot/research/projects/wm_symbolic');B=P/'experiments/expansion_mechanism_analysis_20261021';S=P/'experiments/data_expansion_training_20261021';A=P/'experiments/prediction_error_analysis_20261021';O=P/'experiments/object_recurrent_state_run_20261019';M=P/'experiments/midstart_training_control_20261020';C=P/'experiments/rssm_scaling_run_20261019/cells/roots40'
readhash={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rd(p):readhash[str(p)]=sha(p);return json.loads(p.read_text())
def out(n,r):
 def plain(x):
  if isinstance(x,np.ndarray):return x.tolist()
  if isinstance(x,np.generic):return x.item()
  if isinstance(x,dict):return{k:plain(v)for k,v in x.items()}
  if isinstance(x,(list,tuple)):return[plain(v)for v in x]
  return x
 (B/f'results/{n}.json').write_text(json.dumps(plain(r),ensure_ascii=False,indent=2)+'\n')
 if isinstance(r,list)and r:
  with(B/f'results/{n}.csv').open('w',newline='')as f:
   w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for x in r for k in x)));w.writeheader();w.writerows({k:json.dumps(plain(v),ensure_ascii=False)if isinstance(v,(list,dict))else plain(v)for k,v in x.items()}for x in r)
def condition(skill,c,d):
 if skill not in ['open_drawer','close_drawer']:return'non_drawer_true_stable'if c[3]==d[3]else'non_drawer_GT_change'
 return skill+'_to_'+['closed','middle','open'][d[3]]
def ctx(skill,c,d):return f'{skill}|D{c[3]}>{d[3]}|In{sum(c[:3]==3)}|Held{sum(c[:3]==6)}'
meta={so:{e['episode_id']:e for e in rd(path)}for so,path in [('development',S/'cache/development_metadata.json'),('old_development',O/'cache/development_metadata.json'),('combination_test',S/'cache/combination_test_metadata.json')]};windows=rd(B/'results/matched_windows.json');logs={(r['source'],r['episode_id'],r['step']):r for r in rd(A/'results/execution_log_audit.json')};physical={}
for so,ee in meta.items():
 for eid,e in ee.items():
  for st,r in enumerate(e['rows'],1):
   lab=rd(Path(r['labels_path']));path=Path(r['low_level_log']).parent/'result_private.json';result=rd(path)if path.exists()else{};before=lab['before']['continuous']['drawer_m'];after=lab['after']['continuous']['drawer_m'];c=np.array(lab['before']['categorical']);d=np.array(lab['after']['categorical']);events=result.get('events',[]);physical[so,eid,st]={'source':so,'episode_id':eid,'step':st,'skill':r['skill'],'condition':condition(r['skill'],c,d),'context':ctx(r['skill'],c,d),'before_cat':c.tolist(),'after_cat':d.tolist(),'drawer_before_m':before,'drawer_after_m':after,'distance_to_nearest_drawer_threshold_m':min(abs(after-.025),abs(after-.16)),'status':result.get('status','missing'),'reason':result.get('reason','missing'),'reason_type':(result.get('reason')or'none').rsplit(':',1)[0],'control_steps':result.get('controls',logs[so,eid,st]['controls']),'last_phase':events[-1].get('phase')if events else None,'last_TCP_error_m':events[-1].get('tcp_error_m')if events else None,'declared_intervention':bool(logs[so,eid,st]['declared_controls']),'physical_result_path':str(path),'labels_path':r['labels_path']}
out('physical_outcomes',list(physical.values()));events=[];hazards=[]
for key in sorted({(r['source'],r['seed'],r['arm'])for r in windows}):
 so,seed,arm=key;z=dict(np.load(S/f'results/predictions_{so}_{arm}_seed{seed}.npz'));p=z['prob'];truth=np.c_[z['cat'][:,:3]==6,z['cat'][:,:3]==3];pred=np.c_[p[:,[6,14,22]]>=.5,p[:,[3,11,19]]>=.5];err=np.c_[(pred[:,:3]!=truth[:,:3]).any(1),(pred[:,3:]!=truth[:,3:]).any(1),p[:,24:27].argmax(1)!=z['cat'][:,3]];ix={(str(e),int(o),str(m),int(t)):i for i,(e,o,m,t)in enumerate(zip(z['episode_id'],z['origin'],z['mode'],z['relative_step']))}
 for r in [r for r in windows if(r['source'],r['seed'],r['arm'])==key]:
  eid=r['episode_id'];origin=r['origin'];f=r['free_correct'];first=next((t+1 for t,v in enumerate(f)if not v),-1)
  for t in range(1,4):
   i=ix[eid,origin,'free',t];pr=ix[eid,origin,'free',t-1];o=ix[eid,origin,'one_step',t];ph=physical[so,eid,origin+t];truebefore=int(z['cat'][pr,3]);trueafter=int(z['cat'][i,3]);predbefore=int(p[pr,24:27].argmax());predafter=int(p[i,24:27].argmax());mechanism=ph['condition'];predcorrect=predafter==trueafter
   event={**{k:r[k]for k in ['source','seed','arm','episode_id','origin','initial_relation','future_declared_intervention']},'relative_step':t,'absolute_step':origin+t,'skill':ph['skill'],'condition':ph['condition'],'context':ph['context'],'true_before_drawer':truebefore,'true_after_drawer':trueafter,'pred_before_drawer':predbefore,'pred_after_drawer':predafter,'free_drawer_correct':predcorrect,'observed_drawer_correct':int(p[o,24:27].argmax())==trueafter,'free_joint_correct':f[t-1],'observed_joint_correct':r['observed_correct'][t-1],'free_error_relations':r['free_errors'][t-1],'first_future_joint_error':first==t,'all_previous_future_correct':all(f[:t-1]),'new_drift_on_non_drawer_action':ph['skill']not in ['open_drawer','close_drawer']and truebefore==trueafter and predbefore==truebefore and not predcorrect,'drawer_error_carried_from_before':predbefore!=truebefore and not predcorrect,'p_drawer':p[i,24:27].tolist(),'p_drawer_observed':p[o,24:27].tolist(),**{k:ph[k]for k in ['drawer_before_m','drawer_after_m','distance_to_nearest_drawer_threshold_m','status','reason_type','last_phase','last_TCP_error_m','declared_intervention']}};events.append(event)
   if first==t:hazards.append({**event,'window_initial_correct':r['initial_correct']})
out('drawer_forecasts',events);out('first_future_failure_mechanisms',hazards)
agg=[]
for key in sorted({(r['source'],r['arm'],r['seed'])for r in events}):
 for cond in sorted({r['condition']for r in events if(r['source'],r['arm'],r['seed'])==key}):
  rr=[r for r in events if(r['source'],r['arm'],r['seed'],r['condition'])==(*key,cond)];wrong=[r for r in rr if not r['free_drawer_correct']];agg.append({'source':key[0],'arm':key[1],'seed':key[2],'condition':cond,'window_transition_forecasts':len(rr),'unique_physical_transitions':len({(r['episode_id'],r['absolute_step'])for r in rr}),'free_drawer_correct':sum(r['free_drawer_correct']for r in rr),'observed_drawer_correct':sum(r['observed_drawer_correct']for r in rr),'new_non_drawer_drift':sum(r['new_drift_on_non_drawer_action']for r in rr),'first_joint_error_with_drawer':sum(r['first_future_joint_error']and not r['free_drawer_correct']for r in rr),'wrong_pred_closed':sum(r['pred_after_drawer']==0 for r in wrong),'wrong_pred_middle':sum(r['pred_after_drawer']==1 for r in wrong),'wrong_pred_open':sum(r['pred_after_drawer']==2 for r in wrong),'wrong_mean_p_closed':float(np.mean([r['p_drawer'][0]for r in wrong]))if wrong else None,'wrong_mean_p_middle':float(np.mean([r['p_drawer'][1]for r in wrong]))if wrong else None,'wrong_near_threshold_5mm':sum(r['distance_to_nearest_drawer_threshold_m']<=.005 for r in wrong),'wrong_actual_failed':sum(r['status']=='failed'for r in wrong),'wrong_declared_intervention':sum(r['declared_intervention']for r in wrong)})
out('drawer_mechanism_summary',agg)
# Raw training label coverage and schedule-weighted opportunities at the exact loss targets.
old=rd(O/'cache/train_metadata.json');new=rd(S/'cache/train_metadata.json');eps=old+new;cat=np.concatenate([np.load(O/'cache/train_seed0.npz')['cat'],np.load(S/'cache/train_seed0.npz')['cat']]);valid=np.concatenate([np.load(O/'cache/train_seed0.npz')['cat_valid'],np.load(S/'cache/train_seed0.npz')['cat_valid']]);weights=rd(C/'configs/normalization.json')['class_weights'];unweighted=[];exposures=[];supervision=[];classrows=[];uniqueconds=collections.defaultdict(collections.Counter)
for i,e in enumerate(eps):
 group='old228'if i<228 else'new640'
 for t,r in enumerate(e['rows']):uniqueconds[group][ctx(r['skill'],cat[i,t],cat[i,t+1])]+=1
for group,c in uniqueconds.items():unweighted.extend({'group':group,'context':k,'unique_transitions':v}for k,v in c.items())
for seed in [0,1]:
 for arm,root in [('old228',M),('new868',S)]:
  z=np.load(root/f'configs/schedule_seed{seed}.npz');cc=collections.Counter();sc=collections.Counter();cl=collections.Counter()
  for i,st in zip(z['ids'].flatten(),z['starts'].flatten()):
   e=eps[i];st=int(st);n=e['length']-st;sc['draws']+=1;sc['actual_observation_boundaries']+=n+1;sc['prior_valid_forecasts']+=n;sc['multi_valid_forecasts']+=sum(min(3,n-t)for t in range(n));sc['terminal_valid_targets']+=int(n>2);sc['terminal_recursive_steps']+=max(0,n-2);sc['terminal_length_'+str(max(0,n-2))]+=1
   def label(branch,t):
    if valid[i,t,3]:cl[branch,int(cat[i,t,3])]+=1
   for t in range(st,e['length']+1):label('current',t)
   for t in range(st,e['length']):
    cc[ctx(e['rows'][t]['skill'],cat[i,t],cat[i,t+1])]+=1;label('prior1',t+1)
    for h in range(1,min(3,e['length']-t)+1):label('multi'+str(h),t+h);sc['multi_horizon_'+str(h)]+=1
   if n>2:label('terminal',e['length'])
  exposures.extend({'seed':seed,'arm':arm,'context':c,'effective_transition_draws':n}for c,n in cc.items());supervision.append({'seed':seed,'arm':arm,**dict(sc)})
  for branch in ['current','prior1','multi1','multi2','multi3','terminal']:
   for c in range(3):classrows.append({'seed':seed,'arm':arm,'branch':branch,'drawer_class':c,'valid_target_occurrences':cl[branch,c],'fixed_class_weight':weights[3][c],'weighted_target_occurrences':cl[branch,c]*weights[3][c]})
out('training_unique_contexts',unweighted);out('training_exposure_contexts',exposures);out('supervision_path_counts',supervision);out('drawer_training_class_balance',classrows)
curves=[];curve_summary=[]
for seed in [0,1]:
 for arm,root,name in [('old228',M,'cold_mix'),('new868',S,'new868')]:
  path=root/f'logs/{name}_seed{seed}.csv';readhash[str(path)]=sha(path)
  with path.open()as f:r=[{k:float(v)for k,v in x.items()}for x in csv.DictReader(f)]
  assert len(r)==1500
  for begin,end in [(1,100),(401,500),(901,1000),(1201,1300),(1401,1500)]:
   rr=[x for x in r if begin<=x['update']<=end];row={'seed':seed,'arm':arm,'update_from':begin,'update_to':end,**{k:float(np.mean([x[k]for x in rr]))for k in ['total','current','prior','multi3','terminal_from_h2','outcome','grad_norm']}};curves.append(row)
  rr=r[-500:];curve_summary.append({'seed':seed,'arm':arm,'steps':1500,'seconds':r[-1]['seconds'],'last100_total':np.mean([x['total']for x in r[-100:]]),'last100_mult3':np.mean([x['multi3']for x in r[-100:]]),'last100_terminal':np.mean([x['terminal_from_h2']for x in r[-100:]]),'last500_total_linear_slope_per100updates':float(np.polyfit([x['update']for x in rr],[x['total']for x in rr],1)[0]*100),'last500_multi3_slope_per100updates':float(np.polyfit([x['update']for x in rr],[x['multi3']for x in rr],1)[0]*100),'last500_terminal_slope_per100updates':float(np.polyfit([x['update']for x in rr],[x['terminal_from_h2']for x in rr],1)[0]*100)})
out('training_curve_windows',curves);out('training_convergence_summary',curve_summary);out('additional_raw_input_hashes',readhash)
print('DRAWER_DEV',[r for r in agg if r['source']=='development'and r['condition']in ['non_drawer_true_stable','close_drawer_to_middle']]);print('CONVERGENCE',curve_summary)
