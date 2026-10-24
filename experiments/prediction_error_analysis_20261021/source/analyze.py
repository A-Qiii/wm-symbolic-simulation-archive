from pathlib import Path
import os,sys,json,hashlib,datetime,csv,collections
import numpy as np
P=Path('/home/robot/research/projects/wm_symbolic');A=P/'experiments/prediction_error_analysis_20261021';S=P/'experiments/data_expansion_training_20261021';M=P/'experiments/midstart_training_control_20261020';O=P/'experiments/object_recurrent_state_run_20261019';G=P/'experiments/rgbd_observation_control_20261020'
os.environ['MPLCONFIGDIR']=str(A/'cache/matplotlib');COLORS=['red','blue','pink'];reads={}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def guard(event,args):
 if event=='socket.connect':raise RuntimeError('Offline analysis')
 if event!='open'or not isinstance(args[0],(str,bytes)):return
 p=Path(os.fsdecode(args[0])).resolve();m=args[1];fl=args[2]if len(args)>2 else 0
 writing=(isinstance(m,str)and any(c in m for c in 'wa+'))or(isinstance(fl,int)and fl&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
 if writing and p.is_relative_to(P)and not p.is_relative_to(A):raise RuntimeError('Historical write blocked '+str(p))
sys.addaudithook(guard)
def rd(p):
 p=Path(p);reads.setdefault(str(p),sha(p));return json.loads(p.read_text())
def plain(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return{str(k):plain(v)for k,v in x.items()}
 if isinstance(x,(tuple,list)):return[plain(v)for v in x]
 return x
def save(n,x):
 (A/n).write_text(json.dumps(plain(x),ensure_ascii=False,indent=2)+'\n')
def output(n,rows):
 save('results/'+n+'.json',rows)
 if rows:
  with(A/f'results/{n}.csv').open('w',newline='')as f:
   fields=list(dict.fromkeys(k for r in rows for k in r));w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:json.dumps(plain(v),ensure_ascii=False)if isinstance(v,(list,dict))else plain(v)for k,v in r.items()}for r in rows)
protocol={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Post hoc analysis requested after fixed evaluation. Existing saved predictions and logs only; no training, new data, thresholds or model changes. Test results descriptive only.','source_snapshot_sha256':'cfb229f2df8739775a34719cb50d92e64e245cc125ed2dd6d154f8ec8af3e8bc','endpoint3':'Free relative_step==3, all nonterminal origins with >=3 requests remaining. Joint Held3,In3,drawer at endpoint only. Exact original masks.','new_window3':'Same origins; separately all future1..3 joint valid/correct and all0..3 joint valid/correct; exclude invalid intervals explicitly.','six_step':'Only root origins0 of episodes length6, all0..6; old core length4 separately excluded, never padded as six. First eligible joint error; report interval validity.','first_error_components':'Initial separately. Later wrong Held/In on requested target=target error. Wrong non-target predicate with same before/afterGT=non-target stable error (positive/negative separately); changedGT non-target distinct. Wrong drawer distinct; preserve mixed category instead of priority attribution.','external':'Scan actual requested/applied gripper logs including old role, do not trust new-only prediction disturbance flag. Temporal relation to first error descriptive; verified paired identical inputs and divergentGT can prove unresolved external ambiguity at that boundary.','observed_prefix_diagnostic':'Read saved one_step and current posterior at free first-error step. Changes observation information; not exact GT latent-state reset or causal transition ablation. Development supports diagnosis, holdout description only.','sampling':'Recount saved old/new schedules for fragment draws and effective transitions, context/action coverage; observations do not identify effect of allocation without a new controlled comparison.','no_GPU_or_model_inference':True}
protocol['analysis_source_sha256']=sha(Path(__file__));assert sha(S/'MANIFEST.json')==protocol['source_snapshot_sha256'];save('configs/protocol.json',protocol)
meta={so:rd(path)for so,path in [('old_development',O/'cache/development_metadata.json'),('development',S/'cache/development_metadata.json'),('combination_test',S/'cache/combination_test_metadata.json')]};emd={so:{e['episode_id']:e for e in es}for so,es in meta.items()}
# Actual motor interventions, including historical role episodes omitted by old prediction flags.
logrows=[];faults={};truths={}
for so,es in meta.items():
 for e in es:
  fs=[];ts=[]
  for r in e['rows']:
   controls=rd(r['low_level_log']);label=rd(r['labels_path']);decl=sum(bool(a.get('declared_execution_disturbance',False))for a in controls);override=sum(('requested_gripper'in a and 'applied_gripper'in a and a['requested_gripper']!=a['applied_gripper'])for a in controls)
   if decl:fs.append(int(r['step'])+1)
   ts.append(label['before']['categorical'])
   logrows.append({'source':so,'episode_id':e['episode_id'],'kind':e['kind'],'branch':e['branch'],'branch_mode':e.get('branch_mode',r.get('branch_role','historical_order')),'step':int(r['step'])+1,'skill':r['skill'],'controls':len(controls),'declared_controls':decl,'actual_gripper_overrides':override,'actual_success_label':label['success'],'truth_before':label['before']['categorical'],'truth_after':label['after']['categorical'],'controls_path':r['low_level_log']})
  ts.append(label['after']['categorical']);truths[so,e['episode_id']]=ts;faults[so,e['episode_id']]=fs
output('execution_log_audit',logrows);print('LOGS',len(logrows),flush=True)
allp={};groups={};index={};vv={};counts3=[];windows=[];chains=[];boundary=[];origin_eval=[]
for path in sorted((S/'results').glob('predictions_*.npz')):
 reads[str(path)]=sha(path);z=dict(np.load(path));so=str(z['source'][0]);arm=str(z['arm'][0]);seed=int(z['seed'][0]);key=(so,arm,seed);p=z['prob'];q=np.c_[p[:,[6,14,22]]>=.5,p[:,[3,11,19]]>=.5];y=np.c_[z['cat'][:,:3]==6,z['cat'][:,:3]==3];v=np.c_[z['cat_valid'][:,:3],z['cat_valid'][:,:3]];jv=z['cat_valid'][:,:4].all(1);wrong=(q!=y)&v;dw=(p[:,24:27].argmax(1)!=z['cat'][:,3])&z['cat_valid'][:,3];ok=~(wrong.any(1)|dw)
 allp[key]=z;vv[key]=(q,y,jv,wrong,dw,ok);ix={(str(e),int(o),str(m),int(t)):i for i,(e,o,m,t)in enumerate(zip(z['episode_id'],z['origin'],z['mode'],z['relative_step']))};index[key]=ix;gg=collections.defaultdict(list)
 for i in np.flatnonzero(z['mode']=='free'):gg[str(z['episode_id'][i]),int(z['origin'][i])].append(int(i))
 gg={k:sorted(ids,key=lambda i:int(z['relative_step'][i]))for k,ids in gg.items()};groups[key]=gg
 for (eid,origin),ids in gg.items():
  e=emd[so][eid];assert list(z['relative_step'][ids])==list(range(e['length']-origin+1));fs=faults[so,eid];mode=e.get('branch_mode',e['rows'][0].get('branch_role','historical_order'))
  for h in [1,3,5,6]:
   if len(ids)<=h:continue
   i=ids[h];origin_eval.append({'source':so,'arm':arm,'seed':seed,'episode_id':eid,'origin':origin,'horizon':h,'kind':e['kind'],'branch_mode':mode,'endpoint_eligible':bool(jv[i]),'endpoint_correct':bool(ok[i]),'future_all_valid':bool(jv[ids[1:h+1]].all()),'future_all_correct':bool(ok[ids[1:h+1]].all()),'including_initial_all_valid':bool(jv[ids[:h+1]].all()),'including_initial_all_correct':bool(ok[ids[:h+1]].all()),'future_fault_in_window':any(origin<t<=origin+h for t in fs),'initial_state_after_prior_fault':any(t<=origin for t in fs),'target_combo':bool((z['cat'][i,:3]==3).sum()>=2 and np.isin(z['cat'][i,:3],[3,6]).all())})
  if len(ids)>3:
   ii=ids[:4];windows.append({'source':so,'arm':arm,'seed':seed,'episode_id':eid,'family_id':e['family_id'],'origin':origin,'kind':e['kind'],'branch_mode':mode,'endpoint_eligible':bool(jv[ii[3]]),'endpoint_correct':bool(ok[ii[3]]),'future_all_valid':bool(jv[ii[1:]].all()),'future_all_correct':bool(ok[ii[1:]].all()),'including_initial_all_valid':bool(jv[ii].all()),'including_initial_all_correct':bool(ok[ii].all()),'endpoint_recovered_after_previous_error':bool(ok[ii[3]]and not ok[ii[:3]].all()),'future_fault_in_window':any(origin<t<=origin+3 for t in fs),'prior_fault_visible_at_origin':any(t<=origin for t in fs)})
  if origin!=0 or e['length']!=6:continue
  bad=next((i for i in ids if jv[i]and not ok[i]),None);first=int(z['relative_step'][bad])if bad is not None else-1;components=[];wrongdetail=[];stablepositiveIn=False;stablefalse=False;target_gt_change=False
  if bad is not None:
   if first==0:components=['initial']
   else:
    prev=ids[first-1];target=int(z['target'][bad])
    for k in np.flatnonzero(wrong[bad]):
     ob=k%3;rel='Held'if k<3 else'In';same=bool(y[bad,k]==y[prev,k]);typ='target'if ob==target else'non_target_stable'if same else'non_target_GT_change';components.append(typ)
     if typ=='target'and not same:target_gt_change=True
     if typ=='non_target_stable'and y[bad,k]and rel=='In':stablepositiveIn=True
     if typ=='non_target_stable'and not y[bad,k]:stablefalse=True
     wrongdetail.append({'object':COLORS[ob],'relation':rel,'type':typ,'true_before':bool(y[prev,k]),'true_after':bool(y[bad,k]),'pred_after':bool(q[bad,k])})
    if dw[bad]:components.append('drawer')
   components=sorted(set(components))
  category='none'if bad is None else components[0]if len(components)==1 else'mixed'
  if bad is not None and first>0:
   oneidx=ix[eid,0,'one_step',first];curidx=ix[eid,0,'current',first];oneok=bool(jv[oneidx]and ok[oneidx]);curok=bool(jv[curidx]and ok[curidx])
  else:oneok=curok=None
  chain={'source':so,'arm':arm,'seed':seed,'episode_id':eid,'root_id':e['root_id'],'family_id':e['family_id'],'kind':e['kind'],'branch_mode':mode,'fault_steps':fs,'no_declared_intervention':not fs,'whole_all_valid':bool(jv[ids].all()),'whole_correct':bool(jv[ids].all()and ok[ids].all()),'final_eligible':bool(jv[ids[-1]]),'final_correct':bool(jv[ids[-1]]and ok[ids[-1]]),'first_error_step':first,'first_error_skill':str(z['skill'][bad])if bad is not None else'none','first_error_category':category,'components':components,'wrong_object_relations':wrongdetail,'stable_positive_In_error':stablepositiveIn,'stable_false_error':stablefalse,'target_GT_changed':target_gt_change,'at_declared_fault':first in fs,'after_declared_fault':bool(first>=0 and any(f<first for f in fs)),'before_future_fault':bool(first>=0 and any(f>first for f in fs)),'prediction_file_miss_flag':bool(z['injected_miss_step'][bad])if bad is not None else False,'one_step_observed_prefix_correct_at_free_first_error':oneok,'current_posterior_correct_at_free_first_error':curok}
  chains.append(chain)
  for i in ids:boundary.append({'source':so,'arm':arm,'seed':seed,'episode_id':eid,'step':int(z['relative_step'][i]),'kind':e['kind'],'branch_mode':mode,'no_declared_intervention':not fs,'eligible':bool(jv[i]),'correct':bool(ok[i])})
output('three_step_windows',windows);output('six_step_chains',chains);output('six_step_boundaries',boundary);output('origin_endpoints',origin_eval)
for key in sorted(allp):
 so,arm,seed=key;rr=[r for r in windows if(r['source'],r['arm'],r['seed'])==key]
 counts3.append({'source':so,'arm':arm,'seed':seed,'origins':len(rr),'episodes':len(set(r['episode_id']for r in rr)),'endpoint_eligible':sum(r['endpoint_eligible']for r in rr),'endpoint_correct':sum(r['endpoint_eligible']and r['endpoint_correct']for r in rr),'future_interval_eligible':sum(r['future_all_valid']for r in rr),'future_interval_correct':sum(r['future_all_valid']and r['future_all_correct']for r in rr),'including_initial_interval_eligible':sum(r['including_initial_all_valid']for r in rr),'including_initial_interval_correct':sum(r['including_initial_all_valid']and r['including_initial_all_correct']for r in rr),'endpoint_correct_after_earlier_error':sum(r['endpoint_eligible']and r['endpoint_recovered_after_previous_error']for r in rr)})
output('three_step_counting',counts3)
# Mutually exclusive categories, overlapping components and observation-prefix comparison.
agg=[]
for key in sorted(allp):
 for cohort in ['all','no_declared_intervention','declared_intervention']:
  rr=[r for r in chains if(r['source'],r['arm'],r['seed'])==key and(cohort=='all'or r['no_declared_intervention']==(cohort=='no_declared_intervention'))]
  if not rr:continue
  cc=collections.Counter(r['first_error_category']for r in rr);co=collections.Counter(c for r in rr for c in r['components']);trans=[r for r in rr if r['first_error_step']>0]
  agg.append({'source':key[0],'arm':key[1],'seed':key[2],'cohort':cohort,'episodes':len(rr),'whole_eligible':sum(r['whole_all_valid']for r in rr),'whole_correct':sum(r['whole_correct']for r in rr),'final_correct':sum(r['final_correct']for r in rr),'initial_first':cc['initial'],'target_only_first':cc['target'],'non_target_stable_only_first':cc['non_target_stable'],'drawer_only_first':cc['drawer'],'mixed_first':cc['mixed'],'non_target_GT_change_only_first':cc['non_target_GT_change'],'no_error':cc['none'],'target_component_first':co['target'],'non_target_stable_component_first':co['non_target_stable'],'drawer_component_first':co['drawer'],'first_stable_positive_In':sum(r['stable_positive_In_error']for r in rr),'first_stable_false':sum(r['stable_false_error']for r in rr),'transition_first':len(trans),'first_at_intervention':sum(r['at_declared_fault']for r in rr),'first_after_intervention':sum(r['after_declared_fault']for r in rr),'free_first_corrected_by_observed_prefix':sum(r['one_step_observed_prefix_correct_at_free_first_error']is True for r in trans),'free_first_still_wrong_one_step_but_correct_current':sum(r['one_step_observed_prefix_correct_at_free_first_error']is False and r['current_posterior_correct_at_free_first_error']is True for r in trans),'free_first_wrong_even_current':sum(r['current_posterior_correct_at_free_first_error']is False for r in trans)})
output('six_step_first_error_summary',agg)
# Paired gains/losses, same episodes and same origins, by data group and branch.
paired=[];paired_detail=[]
for so in meta:
 for seed in [0,1]:
  a={(r['episode_id'],r['origin'],r['horizon']):r for r in origin_eval if(r['source'],r['arm'],r['seed'])==(so,'old228',seed)};b={(r['episode_id'],r['origin'],r['horizon']):r for r in origin_eval if(r['source'],r['arm'],r['seed'])==(so,'new868',seed)};assert a.keys()==b.keys()
  for k,r in a.items():
   y=b[k]
   for name,elig,correct in [('endpoint','endpoint_eligible','endpoint_correct'),('future_interval','future_all_valid','future_all_correct'),('including_initial_interval','including_initial_all_valid','including_initial_all_correct')]:
    assert r[elig]==y[elig]
    if not r[elig]:continue
    outcome='both_correct'if r[correct]and y[correct]else'gain'if y[correct]else'loss'if r[correct]else'both_wrong'
    paired_detail.append({'source':so,'seed':seed,'episode_id':k[0],'origin':k[1],'horizon':k[2],'kind':r['kind'],'branch_mode':r['branch_mode'],'metric':name,'outcome':outcome,'future_fault_in_window':r['future_fault_in_window']})
for key in sorted({(r['source'],r['seed'],r['horizon'],r['metric'])for r in paired_detail}):
 rr=[r for r in paired_detail if(r['source'],r['seed'],r['horizon'],r['metric'])==key]
 for factor in ['all','kind','branch_mode','future_fault_in_window']:
  for level in ['all']if factor=='all'else sorted({str(r[factor])for r in rr}):
   tt=rr if factor=='all'else[r for r in rr if str(r[factor])==level];cc=collections.Counter(r['outcome']for r in tt)
   paired.append({'source':key[0],'seed':key[1],'horizon':key[2],'metric':key[3],'factor':factor,'level':level,'eligible':len(tt),**{k:cc[k]for k in ['both_correct','gain','loss','both_wrong']},'new_minus_old_correct':cc['gain']-cc['loss']})
output('paired_same_sample',paired);output('paired_same_sample_details',paired_detail)
# Verify ambiguity from same public feature prefix/requested actions but opposite actual fault outcome.
pairs=[]
for so,es in meta.items():
 byroot=collections.defaultdict(list)
 for i,e in enumerate(es):byroot[e['root_id']].append((i,e))
 for seed in [0,1]:
  path=O/f'cache/development_seed{seed}.npz'if so=='old_development'else S/f'cache/{so}_seed{seed}.npz'
  data=np.load(path);raw=np.load(G/'cache/development_geometry_raw.npz'if so=='old_development'else S/f'cache/{so}_geometry_raw.npz')
  for root,ee in byroot.items():
   disturbed=[(i,e)for i,e in ee if faults[so,e['episode_id']]];normal=[(i,e)for i,e in ee if not faults[so,e['episode_id']]]
   if not disturbed:continue
   assert len(disturbed)==1 and len(normal)==1
   i,e=normal[0];j,f=disturbed[0];t=faults[so,f['episode_id']][0];before=t-1
   cache_fields={k:bool(np.array_equal(data[k][i,:before+1],data[k][j,:before+1]))for k in ['objects','context','availability']};geom_fields={k:bool(np.array_equal(raw[k][i,:before+1],raw[k][j,:before+1]))for k in raw.files};action_same=all(np.array_equal(data[k][i,:t],data[k][j,:t])for k in ['verb','pointer','budget']);exact=all(cache_fields.values())and all(geom_fields.values())and action_same
   c=np.array(truths[so,e['episode_id']][t]);d=np.array(truths[so,f['episode_id']][t]);y1=np.r_[c[:3]==6,c[:3]==3,c[3]];y2=np.r_[d[:3]==6,d[:3]==3,d[3]];different=not np.array_equal(y1,y2)
   for arm in ['old228','new868']:
    key=so,arm,seed;z=allp[key];ix=index[key];u=ix[e['episode_id'],0,'free',t];v=ix[f['episode_id'],0,'free',t];diff=float(np.max(np.abs(z['prob'][u]-z['prob'][v])));ok=vv[key][-1];q={'source':so,'seed':seed,'arm':arm,'root_id':root,'nominal_episode':e['episode_id'],'disturbed_episode':f['episode_id'],'fault_step':t,'public_prefix_features_exact':exact,'cache_fields_equal':cache_fields,'geometry_fields_equal':geom_fields,'requested_action_prefix_exact':action_same,'GT_joint_different_at_fault':different,'prediction_max_abs_diff_at_fault':diff,'predictions_exact':diff==0,'nominal_joint_correct_at_fault':bool(ok[u]),'disturbed_joint_correct_at_fault':bool(ok[v]),'ambiguous_verified':bool(exact and different and diff==0)};pairs.append(q)
output('external_ambiguity_pairs',pairs)
# Exposure: same optimizer budget, actual old/new sampling and transition contexts.
old=rd(O/'cache/train_metadata.json');new=rd(S/'cache/train_metadata.json');oldcat=np.load(O/'cache/train_seed0.npz')['cat'];newcat=np.load(S/'cache/train_seed0.npz')['cat'];eps=old+new;cat=np.concatenate([oldcat,newcat]);exposure=[];actions=[];contexts=[];per_episode=[]
for seed in [0,1]:
 for arm,root in [('old228',M),('new868',S)]:
  f=root/f'configs/schedule_seed{seed}.npz';reads[str(f)]=sha(f);z=np.load(f);ids=z['ids'].flatten();starts=z['starts'].flatten();cc=collections.Counter();ac=collections.Counter();cx=collections.Counter();freq=collections.Counter(ids.tolist())
  for i,st in zip(ids,starts):
   e=eps[i];kind='old_core'if i<120 else'old_role'if i<228 else'new640';cc[kind,'draws']+=1
   for t in range(int(st),e['length']):
    cc[kind,'transitions']+=1;ac[kind,e['rows'][t]['skill']]+=1;c=cat[i,t];cx[kind,f'In{sum(c[:3]==3)}_Held{sum(c[:3]==6)}_drawer{c[3]}']+=1
  for kind in ['old_core','old_role','new640']:
   exposure.append({'seed':seed,'arm':arm,'group':kind,'episode_draws':cc[kind,'draws'],'draw_share':cc[kind,'draws']/len(ids),'effective_transitions':cc[kind,'transitions'],'transition_share':cc[kind,'transitions']/sum(v for(k,m),v in cc.items()if m=='transitions'),'unique_episodes':sum(1 for i in freq if('old_core'if i<120 else'old_role'if i<228 else'new640')==kind)})
  actions.extend({'seed':seed,'arm':arm,'group':k[0],'skill':k[1],'effective_transitions':v}for k,v in ac.items());contexts.extend({'seed':seed,'arm':arm,'group':k[0],'actual_start_context':k[1],'effective_transitions':v}for k,v in cx.items());per_episode.extend({'seed':seed,'arm':arm,'episode_id':eps[i]['episode_id'],'kind':eps[i]['kind'],'draws':freq[i]}for i in range(868 if arm=='new868'else 228))
output('sampling_exposure',exposure);output('sampling_skill_exposure',actions);output('sampling_state_exposure',contexts);output('sampling_episode_draws',per_episode)
# Every protected source snapshot file remains exactly the sealed byte content.
manifest=rd(S/'MANIFEST.json')
for item in manifest['files']:
 p=S/item['path'];assert p.stat().st_size==item['bytes']and sha(p)==item['sha256'],p;assert not(p.stat().st_mode&0o222),p
for p,h in reads.items():assert sha(p)==h,p
save('results/source_reads.json',reads);save('results/verification.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':True,'source_snapshot_files_unchanged':len(manifest['files']),'all_original_source_files_readonly':True,'referenced_JSON_prediction_and_schedule_files_hashed':len(reads),'no_model_inference':True,'no_training':True,'no_collection':True,'no_original_writes':True,'six_step_counts':len(chains),'three_step_windows':len(windows)})
print('COMPLETE',len(chains),len(windows),flush=True)
