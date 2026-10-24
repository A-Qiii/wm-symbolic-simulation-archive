"""Offline replay from saved contacts; no new physics, models or training."""
from pathlib import Path
import json,gzip,hashlib,sys,csv,shutil,collections,datetime
import numpy as np
from official_expansion_fetch import E,R,OLD,read,write
from expansion_stability import evidence,classify,COLORS
P=OLD.parents[1];SHORT=P/'experiments/official_state_resimulation_20261018'
sys.path.insert(0,str(P/'experiments/combination_generalization_20261017/evaluator_review_v1/source'))
from native_evaluator import NativeEvaluator
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def save(name,rows):
 write(E/'results'/f'{name}.json',rows)
 if isinstance(rows,list)and rows:
  keys=list(dict.fromkeys(k for r in rows for k in r))
  with (R/f'{name}.csv').open('w',newline='')as f:
   w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:json.dumps(v,ensure_ascii=False)if isinstance(v,(list,dict))else v for k,v in r.items()}for r in rows)
def main():
 results=read(E/'results/results.json');inputs=read(E/'configs/realized_inputs.json');splits=read(E/'configs/episode_split.json')['splits'];assert len(results)==len(inputs)==256
 for manifest in [E/'configs/frozen_selection.json',E/'configs/frozen_execution.json',OLD/'configs/delivery_manifest.json',SHORT/'configs/delivery_manifest.json']:
  for p,h in read(manifest)['sha256'].items():assert sha(p)==h,p
 budget=read(E/'results/run_budget.json');assert budget['known_completed_physics_ticks']==24576 and budget['stop']is None and budget['retry_count']==0
 ledger=read(E/'results/network_ledger.json');assert ledger['body_bytes_read']==ledger['base_body_bytes_read']+sum(r['body_bytes_read']for r in ledger['requests'])and ledger['body_bytes_read']<=512*1024**2
 assert all(r['status']=='complete'and r['http_status']==206 and r['body_bytes_read']==r['expected_bytes']for r in ledger['requests'])
 assert len({(r['root_id'],r['frame_id'])for r in results})==256 and len({r['root_id']for r in results})==12
 assert all(r['scene']in ['A','B']and r['split']==splits[r['root_id']]for r in results)
 ev={s:NativeEvaluator(s)for s in['A','B']};facts=[];states=[];empty=[];retest=[];motor=[];events=[];early_late=[];physics=obs_count=0
 for index,row in enumerate(results):
  folder=Path(row['folder']);source=np.load(row['source_path'],allow_pickle=False);assert sha(row['source_path'])==row['source_sha256'];audit=read(folder/'control_audit.json');assert audit['control_calls']==12 and np.array_equal(audit['fixed_absolute_command'][0],source['robot_obs'][:3])and np.array_equal(audit['fixed_absolute_command'][1],source['robot_obs'][3:6])and audit['fixed_absolute_command'][2]==source['robot_obs'][-1]
  calls=audit['motor_calls'];assert len(calls)==117;grip=int(source['robot_obs'][-1]);reset={c['jointIndex']:c for c in calls if c['phase']=='reset'};assert len(reset)==9
  for j in range(7):assert reset[j]['targetPosition']==source['robot_obs'][7+j]
  for j in [9,11]:assert reset[j]['targetPosition']==source['robot_obs'][6]/2
  for period in range(12):
   cc={c['jointIndex']:c for c in calls if c['control_period']==period};assert len(cc)==9
   for j in [9,11]:assert cc[j]['targetPosition']==(.04 if grip==1 else 0)and cc[j]['force']==(2 if grip==1 else 200)
  assert read(folder/'integration_ledger.json')['completed_physics_ticks']==96
  motor.append({k:row[k]for k in['root_id','frame_id','split','source_kind']}|{'control_calls':12,'motor_calls':117,'original_gripper_command':grip,'max_arm_target_change_from_original_rad':max(a['max_arm_target_change_rad']for a in audit['period_audits'])})
  history=folder/'physics_history.jsonl.gz';assert sha(history)==row['candidate']['physics_history_sha256']
  with gzip.open(history,'rt')as f:ticks=[json.loads(line)for line in f]
  assert len(ticks)==97
  for t in ticks:
   p=t['physical'];held=[]
   for c in COLORS:
    uid=p['objects']['block_'+c]['uid'];f=[0.,0.];ext=0.
    for contact in p['contacts']:
     if contact[1]==uid:other,link=contact[2],contact[4]
     elif contact[2]==uid:other,link=contact[1],contact[3]
     else:continue
     force=float(contact[9])
     if other==p['robot_uid']:
      for k,family in enumerate([[9,10],[11,12]]):
       if link in family:f[k]=max(f[k],force)
     else:ext=max(ext,force)
    if min(f)>.01 and ext<=.01:held.append('block_'+c)
   assert p['finger_families']==[[9,10],[11,12]]and set(held)==set(p['held']);p['held']=held;lab=ev[row['scene']].evaluate(p);assert lab==t['labels'];t['labels']=lab;cf=evidence(p,lab);assert cf==t['confidence'];t['confidence']=cf;physics+=1
   if t['physics_step']in[32,64,96]:
    op=Path(t['new_observation_path']);assert op.is_relative_to(E/'data/resim')and sha(op)==t['new_observation_sha256'];ob=np.load(op,allow_pickle=False);assert np.array_equal(ob['robot_obs'],t['state']['robot_obs']);assert ob['rgb_static'].shape==(200,200,3)and ob['rgb_gripper'].shape==(84,84,3)and not{'contacts','scene_obs'}&set(ob.files);obs_count+=1
  rebuilt=classify(ticks);cand=read(folder/'candidate_state_label.json');assert cand==row['candidate']
  for k,v in rebuilt.items():assert v==cand[k],(row['frame_id'],k)
  for j,fact in enumerate(cand['fact_order']):
   val=cand['v'][j];y=cand['y'][j];facts.append({k:row[k]for k in['root_id','scene','split','source_kind','frame_id']}|{'fact':fact,'group':'Holding'if j<3 else'In','color':COLORS[j%3],'value':y,'valid':val,'classification':'positive'if val and y else'negative'if val else'unknown','reason':cand['validity_reason'][j],'first_window_failure':cand['diagnostics'][j]['first_window_failure'],'force_stability':cand['diagnostics'][j]['force_stability'],'observation_path':cand['observation_path']})
   early=ticks[8]['labels'];lateclass='positive'if val and y else'negative'if val else'unknown';early_late.append({k:row[k]for k in['root_id','scene','split','source_kind','frame_id']}|{'fact':fact,'group':'Holding'if j<3 else'In','color':COLORS[j%3],'step8_raw_value':early['y'][j],'step8_native_valid':early['v'][j],'step96_classification':lateclass})
   changed=next((t for t in ticks[9:]if t['labels']['y'][j]!=early['y'][j]or not t['labels']['v'][j]),None)
   if changed is not None:events.append({k:row[k]for k in['root_id','scene','split','source_kind','frame_id']}|{'fact':fact,'step8_value':early['y'][j],'first_post8_change_or_invalid_step':changed['physics_step'],'value_at_event':changed['labels']['y'][j],'native_valid_at_event':changed['labels']['v'][j],'confidence_at_event':changed['confidence'][COLORS[j%3]],'drift_at_event':changed['object_drift_from_initial'][COLORS[j%3]],'geometry_at_event':{k:changed['labels']['objects'][COLORS[j%3]][k]for k in['inside','outside','boundary','minimum_side_rim_clearance_m','load_bearing_paths']},'step96_classification':lateclass})
  for state in cand['object_states']:states.append({k:row[k]for k in['root_id','scene','split','source_kind','frame_id']}|state)
  empty.append({k:row[k]for k in['root_id','scene','split','source_kind','frame_id']}|{'empty_hand':cand['empty_hand']})
  if row['source_kind']=='old_source_long_window_retest':
   old=row['old_step8_candidate'];raw8=ticks[8]['labels']
   v8=[all(t['labels']['y'][j]==raw8['y'][j]and t['labels']['v'][j]and t['confidence'][COLORS[j%3]][('Holding'if j<3 else'In')+'_confident']for t in ticks[4:9])for j in range(6)]
   for j,fact in enumerate(cand['fact_order']):
    oldclass='positive'if old['v'][j]and old['y'][j]else'negative'if old['v'][j]else'unknown';newclass='positive'if cand['v'][j]and cand['y'][j]else'negative'if cand['v'][j]else'unknown';fail=next(({'step':t['physics_step'],'raw_value':t['labels']['y'][j],'native_valid':t['labels']['v'][j],'confidence':t['confidence'][COLORS[j%3]],'object_drift':t['object_drift_from_initial'][COLORS[j%3]]}for t in ticks[9:]if t['labels']['y'][j]!=old['y'][j]or not t['labels']['v'][j]),None)
    retest.append({k:row[k]for k in['root_id','scene','split','frame_id']}|{'fact':fact,'old_step8_class':oldclass,'new_step96_class':newclass,'first8_truth_matches_prior':raw8['y'][j]==old['y'][j],'first8_mask_matches_prior':v8[j]==old['v'][j],'first_post8_raw_change_or_invalid':fail,'new_reason':cand['validity_reason'][j]})
  if(index+1)%16==0:print('OFFLINE_RECOMPUTED',index+1,'/256',flush=True)
 save('candidate_facts',facts);save('object_states',states);save('empty_hand_states',empty);save('motor_audits',motor);save('old_source_retest_comparison',retest);save('physical_change_events',events);save('step8_to96_facts',early_late)
 groups=[]
 for rid in sorted(splits):
  rr=[r for r in results if r['root_id']==rid]
  for group in ['Holding','In']:
   ff=[f for f in facts if f['root_id']==rid and f['group']==group];groups.append({'root_id':rid,'scene':rr[0]['scene'],'split':splits[rid],'source_kind':rr[0]['source_kind'],'initializations':len(rr),'group':group,**{k:sum(f['classification']==k for f in ff)for k in['positive','negative','unknown']},'positive_states':len({f['frame_id']for f in ff if f['classification']=='positive'}),'empty_hand_states':sum(r['candidate']['empty_hand']for r in rr)})
 save('episode_summary',groups)
 color=[]
 for split in['all','train_candidate','validation_candidate']:
  for g in ['Holding','In']:
   for c in COLORS:
    ff=[f for f in facts if f['group']==g and f['color']==c and(split=='all'or f['split']==split)];out={'split':split,'group':g,'color':c}
    for cl in ['positive','negative','unknown']:
     subset=[f for f in ff if f['classification']==cl];out[cl]=len(subset);out[cl+'_episodes']=len({f['root_id']for f in subset});out[cl+'_episode_ids']=sorted({f['root_id']for f in subset})
    color.append(out)
 save('color_split_coverage',color)
 state_summary=[]
 for split in['all','train_candidate','validation_candidate']:
  for c in COLORS:
   for st in['Holding','In_drawer','On_table_negative','Other_known_negative','unknown']:
    ss=[s for s in states if s['color']==c and s['state']==st and(split=='all'or s['split']==split)];state_summary.append({'split':split,'color':c,'state':st,'count':len(ss),'episodes':len({s['root_id']for s in ss}),'episode_ids':sorted({s['root_id']for s in ss})})
 save('state_split_coverage',state_summary)
 matched=[];bysegment=collections.defaultdict(list)
 for row in results:
  for hint in row['candidate_hints']:bysegment[hint['annotation_index']].append((hint,row))
 for index,pairs in sorted(bysegment.items()):
  h=pairs[0][0];task=h['task'];target=next((c for c in COLORS if task.startswith('lift_'+c+'_')),None);pairs=sorted(pairs,key=lambda p:p[0]['phase_fraction']);r=pairs[0][1];values=[]
  for h,row in pairs:
   cc=row['candidate'];j=COLORS.index(target)if target else None
   values.append({'phase':h['phase_fraction'],'frame_id':row['frame_id'],'target_Holding_class':('positive'if cc['y'][j]and cc['v'][j]else'negative'if cc['v'][j]else'unknown')if j is not None else None,'all_color_states':cc['object_states'],'empty_hand':cc['empty_hand']})
  matched.append({'annotation_index':index,'root_id':r['root_id'],'split':r['split'],'task':task,'target_color_hint':target,'phases':values,'contains_target_Holding_positive_and_negative':target is not None and {'positive','negative'}<={v['target_Holding_class']for v in values},'not_executed_skill_transition':True})
 save('matched_task_segments',matched)
 metrics=[]
 for step in[32,64,96]:
  for split in['all','train_candidate','validation_candidate']:
   rr=[r for r in results if split=='all'or r['split']==split];ss=[next(t for t in r['checkpoints']if t['physics_step']==step)for r in rr]
   metrics.append({'step':step,'split':split,'initializations':len(rr),'max_object_drift_mm':1000*max(d['translation_m']for t in ss for d in t['object_drift_from_initial'].values()),'max_object_rotation_deg':180/np.pi*max(d['rotation_rad']for t in ss for d in t['object_drift_from_initial'].values()),'max_TCP_drift_mm':1000*max(t['tcp_position_from_initialized_m']for t in ss),'static_MAE':float(np.mean([t['new_RGB_vs_original_MAE_0_255']['rgb_static']for t in ss])),'wrist_MAE':float(np.mean([t['new_RGB_vs_original_MAE_0_255']['rgb_gripper']for t in ss]))})
 save('checkpoint_metrics',metrics)
 basic=[{'split':r['split'],'color':r['color'],'positive_episodes':r['positive_episodes'],'negative_episodes':r['negative_episodes'],'passes':r['positive_episodes']>=2 and r['negative_episodes']>=2}for r in color if r['group']=='Holding'and r['split']!='all']
 save('basic_Holding_coverage',basic)
 save('verification',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':True,'physics_records_recomputed':physics,'new_observations_verified':obs_count,'candidate_fact_cells':len(facts),'motor_calls_verified':sum(x['motor_calls']for x in motor),'initializations':256,'new_source_frames':240,'old_source_retests':16,'source_episodes':12,'train_episodes':sum(s=='train_candidate'for s in splits.values()),'validation_episodes':sum(s=='validation_candidate'for s in splits.values()),'cross_split_episode_leakage':False,'C_D_source_frames':0,'no_new_physics_or_model_forwards':True,'frozen_sources_and_previous_two_seals_unchanged':True,'cumulative_network_body_bytes':ledger['body_bytes_read'],'basic_Holding_coverage_all_cells_pass':all(r['passes']for r in basic),'mask_replay_method':'Recompute Held directly from saved global contacts, native geometry labels from saved physical states, per-tick force evidence and frozen long-window rule; replayed validity implementation, no new simulation.'})
 shutil.copy2(__file__,E/'source'/Path(__file__).name);print('EXPANSION_VERIFIED',physics,obs_count,len(facts),flush=True)
if __name__=='__main__':main()
