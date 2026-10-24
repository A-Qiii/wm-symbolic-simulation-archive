from common import *
import ast,gzip,base64,re
E=P/'experiments/state_predicates_held_down_20261014';B=P/'data/predicate_held_down_replay_20261014';C=P/'experiments/state_predicates_countermotion_20261014';T=P/'experiments/state_predicates_temporal_20261014'
def main():
 protocol=read(E/'configs/protocol.json');checks={}
 for path,h in protocol['frozen_sha256'].items():assert sha(path)==h,path
 previous={}
 for root in [C,T]:
  prior=read(root/'delivery_manifest.json')
  for q in prior['entries']:assert sha(root/q['path'])==q['sha256'],q['path']
  previous[root.name]=prior['files']
 for p in (E/'source').glob('*.py'):ast.parse(p.read_text(),filename=str(p))
 for p in [E/'configs/protocol.json',E/'configs/integration_rule.json',E/'configs/evaluation_inputs_frozen.json',B/'sources_manifest.json']:assert p.stat().st_mode&0o222==0
 checks['frozen_assets']={'protocol_hashes_match':len(protocol['frozen_sha256']),'previous_experiment_files_preserved':previous,'source_AST_valid':True}
 sources=read(B/'sources_manifest.json');manifest=read(B/'data_manifest.json');proposal=read(C/'results/next_step_candidates.json');assert len(sources)==len(manifest['records'])==len(manifest['episodes'])==6;assert [s['original_candidate']for s in sources]==proposal['candidates']
 shapes={'rgb_static':(200,200,3),'rgb_gripper':(84,84,3),'depth_static':(200,200),'depth_gripper':(84,84),'seg_static':(200,200),'seg_gripper':(84,84),'robot_obs':(15,)};allowed=set(shapes)|{f'calib_{cam}_{key}'for cam in ['static','gripper']for key in ['width','height','fov','nearval','farval','viewMatrix','projectionMatrix']}
 frames=imagechecks=statechecks=targetimages=0;counts={k:0 for k in ['prefix_actions','setup_actions','settle_actions','target_actions']}
 for source in sources:
  root=source['root'];ep=B/'episodes'/source['episode_id'];assert read(ep/'source_provenance.json')==source;assert root['environment']=='C'and root['split']=='dev'and not source['new_independent_root'];s=read(ep/'summary.json');v=read(ep/'replay_verification.json');assert s['verification']==v and v['exact_on_all_available_checks']and v['within_predefined_tolerance']and v['no_intermediate_state_reset']and v['max_numeric_error']==0;assert s['physics_guard_passed']and s['eligible_target']
  with np.load(ep/'root_state_private.npz')as a,np.load(Path(source['source_episode'])/'root_reset_private.npz')as b:
   for k in ['robot_obs','scene_obs']:assert np.array_equal(a[k],b[k])
  target=json.load(gzip.open(ep/'target_trace_private.json.gz','rt'));prefix=json.load(gzip.open(ep/'prefix_trace_private.json.gz','rt'));assert len(target)==s['target_recorded_frames']and len(prefix)==s['prefix_actions']
  for segment,sv in zip(source['prefix_segments'],v['segments']):
   sd=Path(segment);control=read(sd/'controls.json');old=json.load(gzip.open(sd/'trace_private.json.gz','rt'));new=target[1:]if segment==source['source_target']else[q for q in prefix if q['segment']==sd.name]
   assert len(new)==len(control)==len(old)==sv['actions'];assert [q['action']for q in new]==control;assert all(x==0 for x in sv['numeric_errors'].values());assert not sv['event_mismatches'];assert sha(sd/'controls.json')==sv['action_sha256']and sha(sd/'trace_private.json.gz')==sv['source_trace_sha256']
   for n,o in zip(new,old):
    for k in ['robot_obs','scene_obs','objects']:assert n[k]==o[k]
   statechecks+=len(new)
  for i,q in enumerate(target):
   assert q['frame']==i
   with np.load(ep/'public'/f'{i:04d}.npz',allow_pickle=False)as z:
    assert set(z.files)==allowed
    for k in z.files:
     assert np.isfinite(z[k]).all()
     if k in shapes:assert z[k].shape==shapes[k]
    assert np.array_equal(z['robot_obs'],q['robot_obs'])
  for q in v['image_checks']:
   assert all(x['bitwise_equal']for typ in ['rgb','mask']for x in q[typ].values());imagechecks+=1
   if q['tag'].startswith(Path(source['source_target']).name+':'):
    tag=q['tag'].split(':')[-1];i=0 if tag=='initial'else len(target)-1 if tag=='final'else int(tag.split('_')[-1])
    with np.load(q['source'])as a,np.load(ep/'public'/f'{i:04d}.npz')as b:
     for cam in ['static','gripper']:
      for typ in ['rgb','seg']:assert np.array_equal(a[typ+'_'+cam],b[typ+'_'+cam])
    targetimages+=1
  r=next(q for q in manifest['records']if q['root_id']==root['root_id']);assert r['anchor_frame']==source['anchor_frame']and r['history_frame_indices']==source['history_frame_indices']==[source['anchor_frame']-d for d in [63,42,21,0]];lab=read(r['labels_private']);assert lab['frames']==[target[i]for i in r['history_frame_indices']];assert lab['physical_definition_unchanged'];window=target[r['anchor_frame']-63:r['anchor_frame']+1];assert len(window)==64 and all(q['objects_audit'][r['target_color']]['held']for q in window);assert max(r['history_frame_indices'])==r['anchor_frame']
  for k in counts:counts[k]+=s[k]
  frames+=len(target)
 assert frames==1744 and imagechecks==128 and targetimages==36 and statechecks==5017 and sum(counts.values())==6709
 checks['recovery']={'existing_C_roots':6,'new_roots':0,'public_frames_all_read_and_finite':frames,'recorded_original_actions_and_state_recompared_exact':statechecks,'executed_action_counts':counts,'original_image_checks_recorded_exact':imagechecks,'target_image_checks_recompared_offline':targetimages,'setup_verification':'Recorded original actions replayed; post-setup full root state and four root history images exact. Original stepwise setup states unavailable.','all_available_numeric_and_event_checks_exact':True,'no_intermediate_reset':True,'future_frames_used_for_input':False}
 frozen=read(E/'configs/evaluation_inputs_frozen.json');rule=read(E/'configs/integration_rule.json')
 for name,key in [('data/inputs.npz','arrays_sha256'),('data/metadata.json','metadata_sha256'),('configs/protocol.json','protocol_sha256'),('configs/integration_rule.json','integration_rule_sha256')]:assert sha(E/name)==frozen[key]
 assert frozen['models']==protocol['models'];meta=read(E/'data/metadata.json');cache=read(E/'results/cache_audit.json');assert all(x==0 for x in cache['parity_max_errors'].values())and cache['only_public_inputs'];assert all(all(q['entity_identity'].values())and all(q['visible_all4'])for q in cache['entity_checks'])
 for r in meta:assert sha(r['feature_path'])==r['feature_sha256']and all(r['visible_all4_by_colour'])
 with np.load(E/'data/inputs.npz')as z:data={k:z[k].copy()for k in z.files}
 assert data['objects'].shape==(6,4,3,780)and data['context'].shape==(6,4,3072)and data['robot'].shape==(6,4,15)and data['v'].all();assert data['y'][:,:3].sum()==6 and data['y'][:,3:6].sum()==1 and data['y'][:,6].sum()==6
 result=read(E/'results/evaluation.json')
 for r,model in zip(result['models'],protocol['models']):
  assert r['seed']==model['seed']and r['checkpoint_sha256']==sha(model['checkpoint'])and r['thresholds']==model['thresholds']and r['inference_batches']==1
  pp=E/'results'/f'seed{r["seed"]}_predictions.npy';assert pp.stat().st_mtime_ns>=(E/'configs/evaluation_inputs_frozen.json').stat().st_mtime_ns>(E/'configs/integration_rule.json').stat().st_mtime_ns;pred=np.load(pp);assert pred.shape==(6,7)and np.isfinite(pred).all();met=evaluate_predictions(pred,data,model['thresholds']);assert met==r['metrics'];assert all(met[g]['FP']==met[g]['FN']==met[g]['unknown']==0 for g in GROUPS);assert r['target_known_TP']==6 and r['target_UNKNOWN']==0
  for i,row in enumerate(r['root_details']):assert row['all_probabilities']==pred[i].tolist()and row['all_truth']==data['y'][i].tolist()
 assert result['prospective_reference_seed']==rule['prospective_reference_seed']==1 and result['reference_meets_predeclared_limited_candidate_rule']and result['new_training_updates']==0 and result['D_or_test_sources']==[]
 checks['frozen_evaluation']={'models':3,'batches_each':1,'predeclared_reference_seed':1,'selection_rule_and_input_freeze_precede_predictions':True,'metrics_recomputed_from_saved_predictions_only':True,'target_known_TP_each':[6,6,6],'held_valid_atoms_each':18,'in_valid_atoms_each':18,'open_valid_atoms_each':6,'errors_or_UNKNOWN_each':[0,0,0],'new_training_or_recalibration':False,'no_D_or_independent_acceptance':True}
 candidate=read(E/'candidate/candidate_manifest.json')
 for a in candidate['assets']:assert sha(E/'candidate'/a['filename'])==sha(a['source'])==a['sha256'];assert (E/'candidate'/a['filename']).stat().st_mode&0o222==0
 contract=E/'candidate/INPUT_CONTRACT.json';assert sha(contract)==candidate['input_contract_sha256'];assert candidate['selection_rule_sha256']==sha(E/'configs/integration_rule.json')and candidate['evaluation_sha256']==sha(E/'results/evaluation.json');assert candidate['model']==rule['reference_model']and not candidate['deployment_performed']and not candidate['independent_acceptance_performed']
 checks['candidate']={'original_seed1_bytes_preserved':True,'input_contract_saved':True,'status':candidate['status'],'runtime_deployment':False}
 old=read(T/'results/evaluation.json');counter=read(C/'results/evaluation.json')
 for r in old['long_history']:
  for condition in ['hard_negative','ordinary_negative','held_positive']:
   q=r['modes']['fixed600']['conditions'][condition]['held'];assert q['FP']==q['FN']==q['unknown']==0
 assert all(r['metrics']['held']['FP']==r['metrics']['held']['unknown']==0 for r in counter['models'])
 olderrors=read(T/'results/remaining_historical_errors.json');assert len(set(q['metadata']['root_id']for q in olderrors))==3 and {q['seed']for q in olderrors}=={0,2}
 motion=read(E/'results/public_motion_audit.json');assert len(motion['windows'])==6 and all(q['max_rise_above_window_start_mm']>6 and not q['monotone_downward']and q['net_TCP_z_mm']<=-20 for q in motion['windows'])
 html=(E/'report.html').read_text();embedded=re.findall(r'data:image/png;base64,([A-Za-z0-9+/=]+)',html);assert len(embedded)==3 and {hashlib.sha256(base64.b64decode(x)).hexdigest()for x in embedded}=={sha(p)for p in (E/'figures').glob('*.png')};assert sha(E/'figures/remaining_old_C.png')==sha(T/'figures/remaining_old_C.png')
 checks['report']={'figures_visually_inspected':3,'embedded_images_match_saved_files':True,'prior_results_referenced_without_new_inference':True,'net_downward_not_monotonic_limit_explicit':True,'old_failure_scenarios_retained':3,'development_not_independent_test_explicit':True,'full_guide_not_completed_explicit':True}
 save(E/'results/delivery_validation.json',{'passed':True,'created_utc':now(),'scope':'Protocol, stored data and artifact validation; saved-metric recomputation only, no repeated model inference.','checks':checks})
 def manifest_for(root,extra=None):
  entries=[]
  for p in sorted(root.rglob('*')):
   if p.is_file()and p.name!='delivery_manifest.json'and '__pycache__'not in p.parts:entries.append({'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':sha(p)})
  out={'root':str(root),'files':len(entries),'bytes':sum(r['bytes']for r in entries),'algorithm':'SHA-256','entries':entries};out.update(extra or{});save(root/'delivery_manifest.json',out);return {'files':out['files'],'bytes':out['bytes'],'manifest_sha256':sha(root/'delivery_manifest.json')}
 bd=manifest_for(B);ed=manifest_for(E,{'dataset_delivery_manifest':str(B/'delivery_manifest.json'),'dataset_manifest_sha256':bd['manifest_sha256']});print(json.dumps({'validated':True,'dataset':bd,'experiment':ed,'replay_frames':frames,'candidate_seed':1},ensure_ascii=False))
if __name__=='__main__':main()
