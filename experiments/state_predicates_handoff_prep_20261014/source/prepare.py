"""Non-destructive manifest repair and public-data availability inventory only."""
from pathlib import Path
from collections import Counter,defaultdict
import ast,json,hashlib,datetime,gzip
import numpy as np
P=Path('/home/robot/research/projects/wm_symbolic');E=P/'experiments/state_predicates_handoff_prep_20261014';T=P/'experiments/state_predicates_temporal_20261014';A=P/'experiments/state_predicates_interface_acceptance_20261014';O=P/'experiments/state_predicates_20261014';HASH={};NPZ_AUDIT={}
KEYS={'rgb_static','rgb_gripper','depth_static','depth_gripper','robot_obs','seg_static','seg_gripper'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
 p=Path(p);b=p.read_bytes();HASH[str(p)]=hashlib.sha256(b).hexdigest();return json.loads(gzip.decompress(b)if p.suffix=='.gz'else b)
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def load(p):
 p=Path(p);HASH[str(p)]=sha(p)
 with np.load(p)as z:return {k:z[k]for k in z.files}
def public(p):
 p=str(p)
 if p not in NPZ_AUDIT:
  d=load(p);assert KEYS<=set(d)and all(k in KEYS or k.startswith('calib_')for k in d);assert d['rgb_gripper'].shape==(84,84,3)and d['robot_obs'].shape==(15,);NPZ_AUDIT[p]={'sha256':HASH[p],'bytes':Path(p).stat().st_size,'rgb_static_shape':list(d['rgb_static'].shape),'rgb_gripper_shape':list(d['rgb_gripper'].shape),'public_keys_only':True}
 return NPZ_AUDIT[p]
def registry_cls():
 p=P/'src/wm_symbolic/vision_cache.py';tree=ast.parse(p.read_text());node=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='VisibleRegistry');ns={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),ns);return ns['VisibleRegistry']
def tensor(f,frames,reg,stats):
 vm,vs,rm,rs=[np.array(stats[k],np.float32)for k in ['vision_mean','vision_std','robot_mean','robot_std']];tok=((f['tokens'].astype(np.float32)-vm)/vs).reshape(4,2,64,384);geo=f['geometry'].reshape(4,2,64,8);valid=f['valid'].reshape(4,2,64);ids=f['visible_registry_ids'].tolist();ob=np.zeros((4,3,2,390),np.float32)
 for j in range(3):
  uid=reg.colour_ids[j];slot=ids.index(uid);sl=slice(slot*4,slot*4+4);ob[:,j,:,:384]=tok[:,:,sl].mean(2)*valid[:,:,slot*4,None];ob[:,j,:,384:389]=geo[:,:,slot*4,:5];ob[:,j,:,389]=geo[:,:,sl,5].mean(2)
 return {'objects':ob.reshape(4,3,780),'context':tok[:,:,48:].reshape(4,2,2,2,2,2,384).mean((3,5)).reshape(4,-1),'robot':(np.stack([x['robot_obs']for x in frames]).astype(np.float32)-rm)/rs}
def repair():
 old=read(T/'data/metadata.json');ca={r['sample_id']:r for r in read(T/'results/cache_audit.json')['details']};arr=load(T/'data/temporal.npz');stats=read(O/'data/normalization.json');fixed=[];mapping=[];regcls=registry_cls()
 for i,r in enumerate(old):
  q=dict(r);sid=r['sample_id'];paths=ca[sid]['long_paths'];inds=[int(Path(p).stem)for p in paths];assert inds==[r['anchor_frame']-d for d in [63,42,21,0]];fp=T/'data/features'/(sid+'.npz');assert sha(fp)==ca[sid]['feature_sha256'];frames=[load(p)for p in paths];reg=regcls();reg.update(frames);f=load(fp);x=tensor(f,frames,reg,stats);err={k:float(np.abs(v-arr['long_history_'+k][i]).max())for k,v in x.items()};assert max(err.values())==0
  lp=read(r['labels_private']);anchor=lp['frames'][-1];assert anchor['frame']==inds[-1];cat=np.array(anchor['labels']['categorical'][:4]);yy=np.r_[cat[:3]==6,cat[:3]==3,cat[3]==2];assert np.array_equal(arr['y'][i],yy);assert r['split']==arr['splits'][i];cv=np.array(anchor['labels']['categorical_valid'][:4]);full=np.r_[cv[:3],cv[:3],cv[3]];assert not np.any(arr['v'][i]&~full)
  for j,c in enumerate(['red','blue','pink']):assert reg.colour_ids[j]==anchor['objects_audit'][c]['object_uid_evaluator_only']
  rgbsha=hashlib.sha256(b''.join(fr[k].tobytes()for fr in frames for k in ['rgb_static','rgb_gripper'])).hexdigest();q.update(history_frame_indices=inds,public_frames=paths,feature_path=str(fp),feature_sha256=sha(fp),public_window_rgb_sha256=rgbsha,metadata_revision='long_history_manifest_v1',input_span_seconds=2.1,control_frequency_Hz=30,source_array={'path':str(T/'data/temporal.npz'),'row_index':i,'keys':{k:'long_history_'+k for k in x}},label_source_contract={'path':r['labels_private'],'use':'frames[-1] current anchor only','anchor_frame':inds[-1],'private_history_frame_indices':[t['frame']for t in lp['frames']],'private_history_is_NOT_long_public_history':True},original_record={'metadata_path':str(T/'data/metadata.json'),'row_index':i,'sha256':hashlib.sha256(json.dumps(r,sort_keys=True).encode()).hexdigest()},original_loss_and_eval_mask_unchanged=arr['v'][i].tolist())
  fields=['history_frame_indices','public_frames','feature_path','feature_sha256','public_window_rgb_sha256'];mapping.append({'sample_id':sid,'row':i,'split':r['split'],'changes':{k:{'before':r.get(k),'after':q[k]}for k in fields if r.get(k)!=q[k]},'tensor_max_abs_error':err,'anchor_label_matches_saved_y':True,'mask_unchanged':True,'feature_matches_original_long_cache_audit':True,'all_registered_ids_match':True,'public_file_sha256':{p:public(p)['sha256']for p in paths}});fixed.append(q)
 assert len(fixed)==54
 save(E/'data/metadata_long_history_v1.json',fixed);save(E/'results/metadata_repair_mapping.json',mapping);save(E/'results/metadata_repair_validation.json',{'passed':True,'records':54,'train':36,'dev':18,'corrected_fields':['history_frame_indices','public_frames','feature_path','feature_sha256','public_window_rgb_sha256'],'maximum_tensor_error':0,'matched_objects_context_robot_scalar_values':sum(arr['long_history_'+k].size for k in ['objects','context','robot']),'labels_and_masks_unchanged':True,'source_array_sha256':sha(T/'data/temporal.npz'),'original_metadata_sha256':sha(T/'data/metadata.json'),'corrected_metadata_sha256':sha(E/'data/metadata_long_history_v1.json'),'original_files_overwritten':False,'no_model_forward':True});print('REPAIRED_AND_VERIFIED_54',flush=True)
def hist(available,t,short_paths=None):
 inds=[t-d for d in [63,42,21,0]];missing=[i for i in inds if i not in available or not Path(available[i]).exists()];paths=[available[i]for i in inds]if not missing else[]
 if paths:
  for p in paths:public(p)
 return {'anchor_control_index':t,'required_control_indices':inds,'complete_2p1s':not missing,'missing_indices':missing,'long_public_paths':paths,'recorded_short_public_paths':short_paths,'new_DINO_cache_required_for_long':bool(paths)}
def inventory():
 transitions=[];by=defaultdict(list);core=P/'data/core1200_v1';manifest=read(core/'data_manifest.json');fi={(r['episode_id'],r['step']):r for r in read(core/'features/index.json')['items']};recover={}
 for s in read(P/'data/predicate_held_down_replay_20261014/sources_manifest.json'):
  d=P/'data/predicate_held_down_replay_20261014/episodes'/s['episode_id'];v=read(d/'replay_verification.json');assert v['exact_on_all_available_checks'];recover[(s['episode_id'],Path(s['source_target']).name)]=d
 for r in manifest['records']:by[r['episode_id']].append(r)
 for eid,rr in sorted(by.items()):
  rr.sort(key=lambda x:x['step']);d=Path(rr[0]['obs_before']).parent.parent;av={k-3:str(d/f'root_observation_{k}.npz')for k in range(4)};t=0;tmp=[]
  for r in rr:
   sd=Path(r['obs_before']).parent;lab=read(r['labels_path']);n=lab['control_samples_including_start']-1;t1=t+n;av[t]=r['obs_before'];av[t1]=r['obs_after']
   for k in range(4):av[t1-3+k]=str(sd/f'boundary_history_{k}.npz')
   if(eid,sd.name)in recover:
    rd=recover[eid,sd.name];pub=sorted((rd/'public').glob('*.npz'));assert len(pub)==n+1
    for sp in [0,n]:
     a=load(rd/'public'/f'{sp:04d}.npz');b=load(r['obs_before'if sp==0 else'obs_after']);assert all(np.array_equal(a[k],b[k])for k in KEYS)
    av.update({t+int(p.stem):str(p)for p in pub})
   tmp.append((r,t,t1));t=t1
  for r,t0,t1 in tmp:
   bef=fi[eid,r['step']];aft=fi[eid,r['step']+1];part='train'if r['split']=='train'else('dev'if r['split']=='dev'else'prior_D');x={'id':f"{eid}__step{r['step']}",'source':'core1200','partition':part,'root_id':r['root_id'],'dev_role':r.get('dev_role'),'skill':r['skill'],'controls':t1-t0,'before':hist(av,t0,bef['public_inputs']),'after':hist(av,t1,aft['public_inputs']),'short_history_both_cached':True,'before_feature_path':bef['path'],'after_feature_path':aft['path'],'private_targets_reference':{'path':r['labels_path'],'before':'before','after':'after','policy':'offline labels only; never model input'},'action_input':{'skill':r['skill'],'declared_budget_steps':r['executor']['budget_steps'],'executor_version':r['executor']['checkpoint']},'model_selection_allowed':part in ['train','dev']};transitions.append(x)
 # Existing supplementary grasp trajectories, with byte-verified original replays as aliases.
 causal=P/'data/predicate_causal_replay_20261014';cr={r['root_id']:r for r in read(causal/'roots_manifest.json')};cm={r['root_id']:r for r in read(causal/'data_manifest.json')['trajectories']};used=set()
 for dn in ['predicate_gap_supplement_20261014','predicate_joint_acceptance_20261014']:
  base=P/'data'/dn;m=read(base/'data_manifest.json')
  for tr in m['trajectories']:
   if tr['branch']not in ['grasp','place']:continue
   rid=tr['root_id'];d=base/'episodes'/tr['trajectory_id'];av={}
   for sample in read(d/'samples_manifest.json'):
    for i,p in zip(sample['history_frame_indices'],sample['public_frames']):av[i]=p
   if rid in cr and cr[rid]['origin']=='replay'and tr['branch']=='grasp':
    rd=causal/'episodes'/(rid+'__grasp');v=read(rd/'replay_verification.json');assert v['exact_on_all_available_checks'];assert cm[rid]['frames']==tr['frames'];av.update({int(p.stem):str(p)for p in(rd/'public').glob('*.npz')});used.add(rid)
   calls=[('lift_'+tr['target_color']+'_block_table',24,tr['frames']-1)]
   if tr['branch']=='place':
    nr=tr['controller_result']['preceding_lift_result']['controls'];calls=[('lift_'+tr['target_color']+'_block_table',24,24+nr),('place_in_drawer',24+nr,tr['frames']-1)]
   for k,(skill,b,e)in enumerate(calls):
    part='train'if tr['split']=='train'else('dev'if tr['split']=='dev'else'prior_D');transitions.append({'id':tr['trajectory_id']+f'__macro{k}','source':'supplement_grasp'if dn.startswith('predicate_gap')else'prior_joint_acceptance','partition':part,'root_id':rid,'skill':skill,'controls':e-b,'before':hist(av,b),'after':hist(av,e),'short_history_both_cached':False,'model_selection_allowed':part in ['train','dev'],'note':'Same A/B grasp replay deduplicated; near_closed_empty is a control probe, not a declared macro skill.'})
 for rid,root in cr.items():
  if rid in used:continue
  assert root['split']=='dev';d=causal/'episodes'/(rid+'__grasp');av={int(p.stem):str(p)for p in(d/'public').glob('*.npz')};last=cm[rid]['frames']-1;transitions.append({'id':rid+'__grasp__macro0','source':'fresh_C_grasp','partition':'dev','root_id':rid,'dev_role':root['dev_role'],'skill':'lift_'+root['target_color']+'_block_table','controls':last-24,'before':hist(av,24),'after':hist(av,last),'short_history_both_cached':False,'model_selection_allowed':True})
 base=P/'data/predicate_interface_acceptance_20261014';m=read(base/'data_manifest.json')
 for root in m['roots']:
  d=base/'episodes'/root['root_id'];av={int(p.stem):str(p)for p in(d/'public').glob('*.npz')}
  for call in root['attempts']:
   if call['kind']!='existing_frozen_skill':continue
   b=call['first_frame']-1;e=call['last_frame'];transitions.append({'id':root['root_id']+f'__macro_t{b:04d}','source':'prior_interface_acceptance','partition':'prior_D','root_id':root['root_id'],'skill':call['request'],'controls':e-b,'before':hist(av,b),'after':hist(av,e),'short_history_both_cached':False,'model_selection_allowed':False,'note':'Previously inspected acceptance; preserve as historical regression only. No outcome/prediction read for this inventory.'})
 summary=[]
 for part,source in sorted(set((x['partition'],x['source'])for x in transitions)):
  rr=[x for x in transitions if x['partition']==part and x['source']==source];bc=sum(x['before']['complete_2p1s']for x in rr);ac=sum(x['after']['complete_2p1s']for x in rr);both=sum(x['before']['complete_2p1s']and x['after']['complete_2p1s']for x in rr);summary.append({'partition':part,'source':source,'macro_transitions':len(rr),'roots':len(set(x['root_id']for x in rr)),'before_2p1_complete':bc,'after_2p1_complete':ac,'both_2p1_complete':both,'need_before_2p1':len(rr)-bc,'need_after_2p1':len(rr)-ac,'short_both_cached':sum(x['short_history_both_cached']for x in rr),'zero_control_calls':sum(x['controls']==0 for x in rr),'skills':dict(Counter(x['skill']for x in rr))})
 windows=[]
 specs=[('core',O/'data/metadata.json',.1),('supplement',P/'experiments/state_predicates_augmented_20261014/data/metadata_train_dev.json',.1),('supplement_D',P/'experiments/state_predicates_augmented_20261014/data/metadata_test.json',.1),('long_repaired',E/'data/metadata_long_history_v1.json',2.1),('held_down_C',P/'experiments/state_predicates_held_down_20261014/data/metadata.json',2.1),('countermotion_C',P/'experiments/state_predicates_countermotion_20261014/data/metadata.json',2.1),('prior_joint',P/'experiments/state_predicates_joint_20261014/data/metadata_acceptance.json',.1),('prior_interface',A/'data/metadata.json',2.1)]
 for source,mp,span in specs:
  meta=read(mp)
  for r in meta:
   part='train'if r['split']=='train'else('prior_D'if r['environment']=='D'else'dev');windows.append({'source':source,'partition':part,'root':r['root_id'],'span_seconds':span,'sample_id':r.get('sample_id',r.get('episode_id','')+f"__boundary{r.get('step',0)}"),'feature_path':r['feature_path']})
 ws=[]
 for part,source in sorted(set((w['partition'],w['source'])for w in windows)):
  rr=[w for w in windows if w['partition']==part and w['source']==source];ws.append({'partition':part,'source':source,'state_windows':len(rr),'roots':len(set(w['root']for w in rr)),'input_span_seconds':rr[0]['span_seconds'],'existing_feature_bytes':sum(Path(w['feature_path']).stat().st_size for w in rr)})
 roots={p:{w['root']for w in windows if w['partition']==p}for p in ['train','dev','prior_D']};assert not roots['train']&roots['dev']and not roots['train']&roots['prior_D']and not roots['dev']&roots['prior_D']
 one=[x for x in transitions if x['source']=='core1200'and x['partition']in ['train','dev']];assert len(one)==960
 for x in one:
  for name in ['before_feature_path','after_feature_path']:x[name+'_sha256']=sha(x[name])
  for end in ['before','after']:
   for p in x[end]['recorded_short_public_paths']:public(p)
 with(E/'data/one_step_short_train_dev_v1.jsonl').open('w')as f:
  for x in one:f.write(json.dumps(x,ensure_ascii=False)+'\n')
 save(E/'data/macro_transition_history_inventory.json',transitions);save(E/'data/state_window_inventory.json',windows);save(E/'results/reuse_summary.json',{'macro_transitions':summary,'state_windows':ws,'distinct_state_roots':{k:len(v)for k,v in roots.items()},'train_dev_test_root_intersections_empty':True,'minimum_existing_short_one_step_pairs':{'train':720,'dev':240,'tune':sum(x['dev_role']=='tune'for x in one),'calibration':sum(x['dev_role']=='calibration'for x in one)},'scope':'Current frozen six-skill core and predicate-study datasets. Legacy pre-core/controller experiments excluded; no unseen data claimed. Raw long public availability is distinct from existing long DINO cache. Macro start is before first action, not a later selected observation.','non_macro_controls_excluded':['near_closed_empty probes','96-step holds','interior ordinary/hard/held state anchors'],'deduplication':'12 A/B causal replays alias12supplement grasps;6C held-down replays enrich existing core transitions rather than add new ones; boundary occurrence counts may share one physical time.','prior_report_erratum':'Previous semantics report prose said26C roots; its machine-readable audit correctly said32. This inventory counts32 distinct C state roots and leaves previous report unchanged.'});print('INVENTORIED_MACROS',len(transitions),summary,flush=True)
def main():
 assert not E.exists(),'Choose a new revision rather than overwrite a prepared artifact'
 for d in ['source','configs','data','results','figures','logs']:(E/d).mkdir(parents=True,exist_ok=True)
 frozen=[T/'data/metadata.json',T/'data/temporal.npz',T/'results/cache_audit.json',T/'source/train.py',T/'source/cache.py',O/'data/normalization.json',A/'delivery_manifest.json',P/'experiments/state_predicates_held_down_20261014/candidate/reader_seed1.pt',P/'experiments/state_predicates_semantics_audit_20261014/delivery_manifest.json']
 save(E/'configs/preparation_scope.json',{'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'authorized':'Create54correctedlongwindowmetadata, before/after mapping, exact cached-input verification, inventory existing macro boundary availability, return finite one-step proposal and compute estimates.','prohibited':['training','model forward','new collection or replay','threshold/decoder/weights changes','test-based model selection','overwriting old records'],'before_sha256':{str(p):sha(p)for p in frozen}});(E/'configs/preparation_scope.json').chmod(0o444)
 repair();inventory();save(E/'results/public_file_checks.json',NPZ_AUDIT);save(E/'results/source_hashes.json',HASH)
 for p,h in read(E/'configs/preparation_scope.json')['before_sha256'].items():assert sha(p)==h,p
 print('PREPARATION_COMPLETE_NO_FORWARD_NO_TRAINING',flush=True)
if __name__=='__main__':main()
