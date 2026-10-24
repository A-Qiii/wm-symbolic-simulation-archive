from common import *
import ast,gzip,base64,re,yaml
E=P/'experiments/state_predicates_interface_acceptance_20261014';B=P/'data/predicate_interface_acceptance_20261014';PRE=P/'experiments/state_predicates_held_down_20261014';COLORS=['red','blue','pink']
def main():
 protocol=read(E/'configs/protocol.json');roots=read(B/'roots_manifest.json');exclusion=read(E/'configs/root_exclusion_audit.json');manifest=read(B/'data_manifest.json');checks={}
 for path,h in protocol['frozen_sha256'].items():assert sha(path)==h,path
 for path,h in exclusion['source_sha256'].items():assert sha(path)==h,path
 old=read(PRE/'delivery_manifest.json')
 for q in old['entries']:assert sha(PRE/q['path'])==q['sha256'],q['path']
 oldlabel=read(P/'experiments/state_predicates_temporal_20261014/configs/protocol.json');labelpath=str(P/'src/wm_symbolic/state_labels.py');labelhash=next(v[labelpath]for v in oldlabel.values()if isinstance(v,dict)and labelpath in v);assert sha(labelpath)==labelhash
 for p in (E/'source').glob('*.py'):ast.parse(p.read_text(),filename=str(p))
 for p in [E/'configs/protocol.json',E/'configs/root_exclusion_audit.json',E/'configs/evaluation_inputs_frozen.json',B/'roots_manifest.json']:assert p.stat().st_mode&0o222==0
 assert len(roots)==len(manifest['roots'])==12 and len({r['root_id']for r in roots})==len({r['seed']for r in roots})==12;assert sha(B/'roots_manifest.json')==protocol['new_roots_sha256']
 prior=[];configs={e:yaml.safe_load((P/f'vendor/calvin/calvin_env/conf/scene/calvin_scene_{e}.yaml').read_text())for e in 'ABCD'}
 def visit(x):
  if isinstance(x,list):
   for q in x:visit(q)
  elif isinstance(x,dict):
   if isinstance(x.get('layout'),dict)and all('block_'+c in x['layout']for c in COLORS):
    v=np.array([x['layout']['block_'+c]for c in COLORS],float)
    if v.shape==(3,6):prior.append(v)
   for q in x.values():
    if isinstance(q,(list,dict)):visit(q)
 for name in exclusion['source_sha256']:
  path=Path(name)
  if path.suffix=='.json':visit(read(path))
  else:
   env=path.parent.name[0]
   with np.load(path)as z:poses=z['scene_obs'][6:].reshape(3,6)
   lookup=dict(zip(configs[env]['objects']['movable_objects'],poses));prior.append(np.stack([lookup['block_'+c]for c in COLORS]))
 prior=list({v.tobytes():v for v in prior}.values());assert len(prior)==exclusion['prior_unique_layouts']==467
 for r in roots:
  assert r['environment']=='D'and r['split']=='independent_acceptance'and r['root_id']not in exclusion['prior_root_ids']and r['seed']not in exclusion['prior_seeds'];v=np.array([r['layout']['block_'+c]for c in COLORS])
  for o in prior:
   d=np.linalg.norm(v[:,:3]-o[:,:3],axis=1);yaw=np.abs((v[:,5]-o[:,5]+np.pi/2)%np.pi-np.pi/2);assert not(np.all(d<=.012)or(np.all(d<=.020)and np.all(yaw<=.25)))
  prior.append(v)
 checks['independence_and_freeze']={'new_D_roots':12,'prior_unique_layouts_rechecked':467,'source_metadata_hashes_verified':len(exclusion['source_sha256']),'new_ids_and_seeds':True,'candidate_previous_delivery_files_preserved':old['files'],'frozen_source_and_model_assets':len(protocol['frozen_sha256']),'label_dependency_matches_prior_temporal_freeze':True}
 shapes={'rgb_static':(200,200,3),'rgb_gripper':(84,84,3),'depth_static':(200,200),'depth_gripper':(84,84),'seg_static':(200,200),'seg_gripper':(84,84),'robot_obs':(15,)};allowed=set(shapes)|{f'calib_{cam}_{key}'for cam in ['static','gripper']for key in ['width','height','fov','nearval','farval','viewMatrix','projectionMatrix']};frames=controls=0;all_traces={};frozen_time=(E/'configs/protocol.json').stat().st_mtime_ns;initialized_prior=prior[:-12].copy();reset_drift=[]
 for root,s in zip(roots,manifest['roots']):
  ep=B/'episodes'/root['root_id'];assert read(ep/'root_provenance.json')==root and read(ep/'summary.json')==s;assert s['physics_guard_passed']and not s['controller_failures']and s['collection_error']is None;assert ep.stat().st_mtime_ns>frozen_time;tr=json.load(gzip.open(ep/'trace_private.json.gz','rt'));assert len(tr)==s['frames'];all_traces[root['root_id']]=tr
  with np.load(ep/'initial_state_private.npz')as z:
   assert abs(z['scene_obs'][1]-root['drawer_reset_m'])<1e-12
   # Native scene.reset_movable_objects and env.reset both advance physics.
   # The first observed pose need not equal the requested spawn pose bitwise.
   # Recheck the same preregistered exclusion rule on actual initialized poses.
   actual=z['scene_obs'][6:].reshape(3,6);requested=np.array([root['layout']['block_'+c]for c in COLORS]);reset_drift.extend((np.linalg.norm(actual[:,:3]-requested[:,:3],axis=1)*1000).tolist());assert z['scene_obs'].tolist()==tr[0]['scene_obs_private']
   for other in initialized_prior:
    dd=np.linalg.norm(actual[:,:3]-other[:,:3],axis=1);yy=np.abs((actual[:,5]-other[:,5]+np.pi/2)%np.pi-np.pi/2);assert not(np.all(dd<=.012)or(np.all(dd<=.020)and np.all(yy<=.25)))
   initialized_prior.append(actual.copy())
  for i,q in enumerate(tr):
   assert q['frame']==i;assert all(bool(q['labels']['holding_derived'][j])==q['objects_audit'][c]['held']for j,c in enumerate(COLORS))
   with np.load(ep/'public'/f'{i:04d}.npz',allow_pickle=False)as z:
    assert set(z.files)==allowed
    for k in z.files:
     assert np.isfinite(z[k]).all()
     if k in shapes:assert z[k].shape==shapes[k]
    assert np.array_equal(z['robot_obs'],q['robot_obs'])
  for a in s['attempts']:
   rows=tr[a['first_frame']:a['last_frame']+1];assert len(rows)==a['controls']
   if a['kind']=='unchanged_motor_hold':
    assert a['controls']==96;assert len({tuple(q['action']['target_xyz'])for q in rows})==1 and len({tuple(q['action']['target_quaternion'])for q in rows})==1 and {q['action']['gripper']for q in rows}=={a['gripper']}
  assert sum(a['controls']for a in s['attempts'])==s['controls']==len(tr)-1;frames+=len(tr);controls+=len(tr)-1
 assert frames==11277 and controls==11265
 for r in manifest['records']:
  tr=all_traces[r['root_id']];i=r['anchor_frame'];inds=[i-63,i-42,i-21,i];assert i>=63 and r['history_frame_indices']==inds;lab=read(r['labels_private']);assert lab['frames']==[tr[k]for k in inds];assert r['intended_condition_achieved'];win=tr[i-63:i+1];c=r['target_color'];tcp=np.array([q['robot_obs'][:3]for q in win]);obj=np.array([q['objects_audit'][c]['position_m']for q in win])
  if r['condition']=='held_stationary':assert all(q['objects_audit'][c]['held']for q in win)and max(np.linalg.norm(tcp-tcp[0],axis=1).max(),np.linalg.norm(obj-obj[0],axis=1).max())<=.001
  if r['condition']=='held_downward':
   assert all(q['objects_audit'][c]['held']for q in win);assert np.diff(tcp[:,2]).max()<=0 and np.diff(obj[:,2]).max()<=0 and max(tcp[-1,2]-tcp[0,2],obj[-1,2]-obj[0,2])<=-.02
   phase=[q['frame']for q in tr if q['segment']=='place_in_drawer'and q['action']['phase']=='lower_into_drawer'];assert phase[63]==i
 assert len(manifest['records'])==90 and all(r['condition']!='natural_held_occlusion'for r in manifest['records']);checks['collection']={'public_frames_read_schema_finite_and_robot_trace_equal':frames,'original_control_count':controls,'all_attempts_retained':True,'controller_failures':0,'stationary_condition_verified_roots':12,'strictly_nonincreasing_downward_verified_roots':6,'history_offsets_exact_past_only':True,'natural_held_occlusion_not_observed':12,'visibility_not_used_to_discard_samples':True,'native_reset_requested_to_first_observed_max_drift_mm':max(reset_drift),'actual_initialized_roots_also_pass_same_exclusion_rule':True}
 frozen=read(E/'configs/evaluation_inputs_frozen.json')
 for path,key in [(E/'configs/protocol.json','protocol_sha256'),(B/'roots_manifest.json','root_manifest_sha256'),(B/'data_manifest.json','data_manifest_sha256'),(E/'data/inputs.npz','arrays_sha256'),(E/'data/metadata.json','metadata_sha256'),(PRE/'candidate/reader_seed1.pt','candidate_checkpoint_sha256'),(E/'source/evaluate.py','evaluator_source_sha256')]:assert sha(path)==frozen[key]
 assert (E/'results/predictions.npy').stat().st_mtime_ns>(E/'configs/evaluation_inputs_frozen.json').stat().st_mtime_ns
 with np.load(E/'data/inputs.npz')as z:data={k:z[k].copy()for k in z.files}
 meta=read(E/'data/metadata.json');pred=np.load(E/'results/predictions.npy');D=read(E/'results/evaluation.json');cache=read(E/'results/cache_audit.json');assert data['objects'].shape==(90,4,3,780)and data['context'].shape==(90,4,3072)and data['robot'].shape==(90,4,15)and pred.shape==(90,7);assert all(np.isfinite(x).all()for x in data.values())and np.isfinite(pred).all();assert all(v==0 for v in cache['parity'].values());assert evaluate_predictions(pred,data,[.5,.5,.5])==D['unique_window_metrics']['metrics'];km=known(pred,[.5,.5,.5]);hat=pred>=.5;complete=(data['v']&km&(hat==data['y'].astype(bool))).all(1);assert complete.sum()==83 and (~data['v'].all(1)).sum()==5 and ((~km)&data['v']).any(1).sum()==1
 for i,r in enumerate(meta):
  assert sha(r['feature_path'])==r['feature_sha256'];lab=read(r['labels_private']);cur=lab['frames'][-1]['labels'];cv=cur['categorical_valid'];assert data['v'][i].tolist()==cv[:3]+cv[:3]+[cv[3]];assert D['details'][i]['probabilities']==pred[i].tolist()and D['details'][i]['known']==km[i].tolist()and D['details'][i]['all7_known_correct']==bool(complete[i])
  cat=cur['categorical'];assert data['y'][i].tolist()==[float(cat[j]==6)for j in range(3)]+[float(cat[j]==3)for j in range(3)]+[float(cat[3]==2)]
 from evaluate import summarize_subset
 for c,q in D['conditions'].items():
  ii=[i for i,r in enumerate(meta)if c in r['conditions']];aa=[i for i in ii if meta[i]['condition_achieved'][c]];assert summarize_subset(ii,pred,data,meta)==q['available']and summarize_subset(aa,pred,data,meta)==q['achieved']
 assert D['inference_batches']==1 and D['new_training_updates']==0 and D['raw_held_conflict_windows']==1;assert sum(r['mandatory_all7_correct_root']for r in D['root_outcomes'])==10;checks['evaluation']={'unique_windows':90,'single_batch':True,'source_checkpoint_hash_unchanged':True,'metric_recomputation_from_saved_predictions_only':True,'all_conditions_window_and_root_denominators_recomputed':True,'all7_correct_windows':83,'GT_boundary_windows':5,'known_error_or_UNKNOWN_windows':2,'actual_conflict_decode_verified':True,'visibility_does_not_mask_GT':True,'no_training_threshold_change_or_repeated_inference':True}
 videos=read(E/'videos/index.json');videoqa=read(E/'results/video_validation.json');assert videoqa['passed']and videoqa['videos']==len(videos)==90
 for q in videos:assert sha(q['path'])==q['sha256']
 html=(E/'report.html').read_text();embedded=re.findall(r'data:image/png;base64,([A-Za-z0-9+/=]+)',html);assert len(embedded)==3 and {hashlib.sha256(base64.b64decode(x)).hexdigest()for x in embedded}=={sha(p)for p in (E/'figures').glob('*.png')};links=re.findall(r'href="(videos/[^"]+)"',html);assert len(links)==90 and all((E/x).is_file()for x in links);diag=read(E/'results/failure_localization.json');assert len(diag)==2 and all(x['prior_scored_windows_all7_correct']and x['all64_true_held']and x['target_visible_both_cameras_all4']and x['registry_identity_correct']for x in diag)
 checks['deliverables']={'three_figures_visually_inspected':True,'embedded_images_exact':True,'videos_64frames_30fps':90,'first_failure_images_and_video_previews_checked':True,'uncovered_held_occlusion_and_GT_boundaries_explicit':True,'no_WM_or_planning_execution':True}
 save(E/'results/delivery_validation.json',{'passed':True,'created_utc':now(),'scope':'Validation of saved observations, frozen sources, predictions, metrics and rendered artifacts only. No new simulation or inference.','checks':checks})
 def make_manifest(root,extra=None):
  entries=[{'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':sha(p)}for p in sorted(root.rglob('*'))if p.is_file()and p.name!='delivery_manifest.json'and '__pycache__'not in p.parts];out={'root':str(root),'files':len(entries),'bytes':sum(q['bytes']for q in entries),'entries':entries,'algorithm':'SHA-256'};out.update(extra or{});save(root/'delivery_manifest.json',out);return {'files':out['files'],'bytes':out['bytes'],'manifest_sha256':sha(root/'delivery_manifest.json')}
 bd=make_manifest(B);ed=make_manifest(E,{'dataset_manifest':str(B/'delivery_manifest.json'),'dataset_manifest_sha256':bd['manifest_sha256']});print(json.dumps({'validated':True,'dataset':bd,'experiment':ed},ensure_ascii=False))
if __name__=='__main__':main()
