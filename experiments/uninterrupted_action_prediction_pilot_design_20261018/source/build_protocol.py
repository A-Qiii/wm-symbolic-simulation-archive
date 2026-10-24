"""Protocol metadata only. No simulator, renderer, ML framework or model imports."""
from pathlib import Path
import json,csv,hashlib,random,collections,datetime,shutil
P=Path('/home/robot/research/projects/wm_symbolic');W=Path('/home/robot/Documents/Codex/2026-10-13/task-4')
NAME='uninterrupted_action_prediction_pilot_design_20261018';E=P/'experiments'/NAME;R=P/'reports'/NAME
def read(p):return json.loads(Path(p).read_text())
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def csvwrite(p,rows):
 with p.open('w')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
 assert not E.exists()and not R.exists()
 for x in ['configs','source','results','logs']:(E/x).mkdir(parents=True,exist_ok=True)
 R.mkdir(parents=True)
 scenes=['A','B','C'];colors=['red','blue','pink'];geometries=['aligned','offset_localY_12mm'];starts=['closed_near_table','empty_grasp_high','grasp_attempt_high'];actions=['hold_closed','lift_closed','hold_open']
 rng=random.Random(261020900);rowperm=list(range(3));colperm=list(range(3));symbols=actions.copy();offsets=[0,0,1,1,2,2]
 for a in [rowperm,colperm,symbols,offsets]:rng.shuffle(a)
 roots=[]
 for si,scene in enumerate(scenes):
  for ci,color in enumerate(colors):
   for gi,geometry in enumerate(geometries):
    for pi,start in enumerate(starts):
     k=len(roots);action=symbols[(rowperm[si]+colperm[pi]+offsets[ci*2+gi])%3]
     roots.append({'root_id':f'obs54_{scene}_{k:02d}','layout_seed':261020800+k,'scene':scene,'split':'train'if scene in['A','B']else'dev','target_color_private':color,'geometry_private':geometry,'preparation_private':start,'assigned_action_private':action,'parent_root':None,'collection_status':'NOT_STARTED'})
 order=list(range(54));random.Random(261020901).shuffle(order)
 for rank,k in enumerate(order,1):roots[k]['execution_rank']=rank
 assert len(roots)==54
 for s in scenes:
  part=[r for r in roots if r['scene']==s];assert len(part)==18
  for key,values,count in [('target_color_private',colors,6),('geometry_private',geometries,9),('preparation_private',starts,6),('assigned_action_private',actions,6)]:
   assert collections.Counter(r[key]for r in part)==dict.fromkeys(values,count)
  for st in starts:
   for ac in actions:assert sum(r['preparation_private']==st and r['assigned_action_private']==ac for r in part)==2
 for c in colors:
  for g in geometries:
   for st in starts:
    for ac in actions:assert sum(r['target_color_private']==c and r['geometry_private']==g and r['preparation_private']==st and r['assigned_action_private']==ac for r in roots)==1
 old=P/'experiments/holding_action_fork_feasibility12_20261018';registry=read(old/'configs/historical_exclusion.json');known_seeds=set(registry['old_seeds']);known_ids=set(registry['old_root_ids'])
 for name in ['holding_action_fork_feasibility12_20261018','relation_GT_boxes36_ABC_20261018','acquisition_scene_surface_pilot8_20261018']:
  prev=read(P/'experiments'/name/'configs/protocol.json')['roots'];known_seeds|={r['seed']for r in prev};known_ids|={r['root_id']for r in prev}
 assert not known_seeds&{r['layout_seed']for r in roots};assert not known_ids&{r['root_id']for r in roots}
 csvwrite(E/'configs/assignments.csv',roots);shutil.copy2(E/'configs/assignments.csv',R/'assignments.csv')
 protocol={
 'status':'DESIGN_ONLY_NOT_AUTHORIZED_FOR_EXECUTION_THIS_TURN','version':1,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'objective':'Observational/predictive action-conditioned Holding from uninterrupted independent-root physical rollouts. Exact same-state counterfactual replay is not required.',
 'closed_route':{'name':'exact native snapshot fork restoration','final_negative_manifest_sha256':'5cddca9456f27c68c3e50f5e93252310f327ea3f35eb231b86ddf3c9ca9e871d','new_restore_debugging':False,'historical_outputs_unchanged':True},
 'budget':{'roots':54,'one_initialization_per_root':True,'actions_per_root':1,'action_controls':72,'physics_ticks_per_control':8,'physics_hz':240,'action_seconds':2.4,'root_controls_max':900,'preparation_controls_max_including_settle':828,'primary_transitions_max':54,'dual_camera_saved_states_max':108,'saved_states_per_root_max':2,'root_replacements':0,'grasp_retries':0,'post_run_translate_or_return':False,'new_downloads':0,'new_model_fits_this_turn':0,'new_physics_steps_this_turn':0,'new_render_calls_this_turn':0,'first50_saved_states_storage_audit':True,'first50_completed_macrotransitions_storage_audit_if_reached':True,'128_transition_check_reached':False,'GPU':'single queue using existing runs/gpu.lock'},
 'split':{'train':'A18+B18=36 independent roots','dev':'C18; exploratory known environment, no finaltest','test':'none; D untouched','unit':'root, all views/queries/labels/states inherit root','no_legacy_data_merge':True},
 'assignment':{'method':'randomized balanced Latin assignment; constraints induce dependence between assignments, not outcome dependence','seed':261020900,'layout_seeds':list(range(261020800,261020854)),'execution_order_seed':261020901,'rows_scene_permutation':rowperm,'columns_preparation_permutation':colperm,'action_symbol_permutation':symbols,'block_offsets_by_color_then_geometry':offsets,'generated_before_physics_or_labels':True,'assignment_file_sha256':sha(E/'configs/assignments.csv'),'no_action_replacement_on_refusal':True},
 'layout':{'sampler':'reuse scene-native table x fractions .18/.50/.82 with +/- .012 jitter, independently permuted colors, y[-.115,-.055],z.46,yaw[-.20,.20]; separate per-root RNG from action RNG','scene_table_bounds':{s:next(r['native_table_surface']for r in read(old/'configs/protocol.json')['roots']if r['scene']==s)for s in scenes},'drawer_initial_q_m':0.005,'note':'fixed closed drawer in all roots to avoid an irrelevant extra factor; unchanged native configuration value used in prior accepted/failed roots','registered_root_id_and_seed_overlap':False,'future_layout_signature_near1mm_and_RGB_checks':'must compare against registered historicalABCmetadata before model use; an initialization collision stops manifest/collection, never replacement sampling','initial_layouts_generated_this_turn':False},
 'preparation':{'candidate':'PhysicalOracleV2.plan_lift deterministic first ranked geometricallyvalid candidate; no controller.run() or outcome-based retry','geometry':{'aligned':'unchanged candidate waypoints','offset_localY_12mm':'shift above/grasp/lift waypoints +.012m along grasp quaternion localY; raise waypoint unchanged'},'closed_near_table':'raise/above/grasp open, close20; measure actual labels','empty_grasp_high':'raise/above/directly original lift waypoint open, close20; never approach object; actual label not forced','grasp_attempt_high':'raise/above/grasp open, close20, original lift waypoint closed regardless measuredHolding; only safety may stop','common_high_anchor':'same original candidate lift waypoint, current source target initial z+.14m; do not use preparation label as predictor input','settling':'native initial24open controls plus unchanged move settling; high moves use speed.0025 and settle18; emptyhigh closes20 after arrival, graspattempt closes20 near table before high move; no post-outcome adaptive waiting','planned_support_labels':None},
 'actions':{'origin':'actual pre-action public TCP and orientation; quaternion conversion specified by currentEuler observation; program fixed at t0','hold_closed':'72 absoluteTCP holds, grip-1','lift_closed':'12hold,48worldZ+.0025m/control,12holdat+.12m;grip-1','hold_open':'72 absoluteTCP holds,grip+1','actual_command_log':'xyz3,quaternion4,grip1; exactcheck against precommitted72program','IK_motor_targets':'private audit only; depend on intermediate state, never model input','early_abort':'capture lastactualstate and storeactualexecutedcommands/duration asfailureaudit; no72target and nofixedhorizontraining row; nofalse label'},
 'public_predictor_input_allowlist':{'rgb_static_t0':'uint8 200x200x3','rgb_gripper_t0':'uint8 84x84x3','robot_obs_t0':'15 nativeEuler-mode values:TCP3,orientation3,width1,joints7,currentgrip1','action_program':'72x8 fixed-at-t0 actualTCPcommands, not actionname','duration_seconds':2.4},
 'forbidden_predictor_inputs':['post_action_RGB_or_features','contacts','scene_obs','simulator poses','GT boxes/masks','depth/calibration in this pilot','task_oracle_success','preparation/phase/index','root/seed/split/file-name features','target-color indicator','early-stop realized duration or reason','future adaptive IK commands'],
 'targets':{'primary':'three fixed color queries Holding atactualt72; preHolding onlylabelsforpersistence/statification, neverGTfutureinput','post_RGB':'separate synchronizedtarget/archive, never input','Holding':'bilateral finger-family forces withnoexternal support; cutoffs .005/.01/.015N, last12controls stable, no multiHolding contradiction','SupportTable':'private directtarget-to-tablebase load support:force>.01N,normal_z>.5,gap<=.003m; uncertain if .005/.015N or .45/.55normalclassification differs, requirelast12stability; notIn','visibility':'atleastone camera atpre andpost with target andeachfingerfamily >=4pixels; allinvisible/unstable retainedunknown','hard_negative':'targetHoldingFalse, bothfingerfamilies in stableloadcontact AND tableupwardsupport; observable; countdistinctrootonce, never highempty/non-target proxies'},
 'safety':{'unchanged_thresholds':{'moving_device_clearance_m':0.012,'other_non_target_clearance_m':0.002,'shadow_IK_error_m':0.018,'unrequested_device_drift':0.003,'non_target_displacement_m':0.01,'robot_penetration_stop_m':0.01,'non_target_robot_contact_stop_N':0.01},'no_actual_world_mutation_after_initialization':['resetSimulation','resetJointState','resetBasePositionAndOrientation','resetBaseVelocity','restoreState','loadBullet','createConstraint','removeConstraint','changeConstraint','changeDynamics','setCollisionFilterPair','setCollisionFilterGroupMask','applyExternalForce','applyExternalTorque','setGravity','setPhysicsEngineParameter'],'shadow_client_only':'existing query reset exemption; no actualclient resets','new_wrapper_required':'fixed9finger-opening samples in shadow client with originalmargins, because existing check sync retains currentfingerwidth; do notclaim alreadyvalidated','safety_refusal':'retainassignedroot/action, noalternateaction/candidate, noresample','stop_rule':'infrastructure/integrity/forbiddenmutation stopsbatch; ordinarygeometry/safetyfailure stopsroot, preserves it, nextprespecifiedroot maycontinue'},
 'data_gate':{'all_integrity_checks':True,'completed_safe_stable_visible_target_roots':{'train_min':24,'dev_min':12},'each_color_future_target_class_roots':{'train_positive_min':2,'train_negative_min':2,'dev_positive_min':1,'dev_negative_min':1},'each_direction_0to1_and_1to0':{'train_min':2,'dev_min':1,'colors_total_min':2},'unchanged_each_0to0_and_1to1':{'train_min':2,'dev_min':1},'each_action_hold_closed_and_lift_closed':{'train_positive_min':3,'train_negative_min':3,'dev_positive_min':2,'dev_negative_min':2,'colors_per_outcome_across_all_min':2},'each_color_target_supported_hard_negative':{'train_min':1,'dev_min':1,'anchor':'pre orpost, distinctrootonce'},'positive_counts_observed':None,'on_failure':'close fixedpilot as insufficientcoverage; noextra roots/offset/tolerance/labels/modeltraining; not proof no dynamics information exists'},
 'future_baselines_not_executed':{'arms':['learned_current_state_persistence','state_only','action_only','state_plus_action'],'feature_boundary':'frozen cachedDINOv2 fullframes, olddataindependent384x128projection eachcamera; noGTcrops; state256+15=271, action72*8+1=577','head':'same848->32->3 SiLU,27267parameters; unavailableinputblocks zeroedafter sharedtrain-onlynormalization','persistence':'learncurrentt0Holding ontrainrootlabels, carryprobabilityto72; neveruseGTinitiallabelasinput','other_heads':'predictfuturet72Holding','seeds':[0,1],'updates_each':600,'sampler':'sameprecommitteduniformtrainrootbatch32 schedule acrossarms; noCdrivenselection','loss':'masked per-root BCE across3colors with fixedweights .5assignedtarget+.25eachnontarget, renormalized over observed labels; unknowns neverFalse; targetflag onlyselectsloss/outputscoring, not input; nooutcomehard-negativeoversampling','optimizer':'AdamW lr.001 weight_decay.001 clip1; fixedfloat32,TF32off','checkpoint':'final600 only,threshold.5','training_permission':'not authorized in thisdesignturn; noautomaticfitaftercoveragegate'},
 'future_evaluation':{'primary':'targetHolding NLL/BA;0to1/1to0recall;target-supported-negativeFPR; failure/unknowncoverage','strata':['action','measuredpreHolding/support','targetcolor','geometry','preparation','scene','highempty_vs_highgrasp','closed-actions-only'],'nontarget':'separate, cannotdominateprimaryscore','uncertainty':'pairedwholedevrootbootstrap2000; allviews/queriesofrootresampletogether; undefined=null','joint_gain_rule':'bothseeds state+action targetNLL better thaneachofthreebaselines with95%CIupper<0; also improvementwithinhold_closed andlift_closed subsets; notopen-onlygain','claim_limit':'multimodalconditionalprediction, notRGB-specificgain orindividualcausal effects'},
 'causal_claims':{'unavailable':['exactindividualsame-statecounterfactual','deterministicactionprecondition/effectrules','specificinternalsolvermechanism','RSSMmulti-steporTS/LTLfclosedloopcompetence'],'available_to_test':['out-of-rootconditionalpredictionunderfixedsampling/executor','incrementalvalueofpre-observationandaction','calibration/errorcoverageandfailurepatterns'],'randomization_caveat':'intent-to-treatpopulationcontrastsrequirecompleteoutcome/censoringassumptions; primarycompletevalidsubset may beselectedbysafety/visibility, so notclaimedunbiasedATE'},
 'required_implementation_before_any_collection':['newforward-onlycollector usingrelation36MotorAdapter; donotsubclassv3Runtime','noctl.run(),noheld-conditionedrolloutselection,nooutcome-basedretry','strictpublicinputwriter separatefromprivateannotations/postRGB','extra denyrestore/loadBulletguard afterinitialization','fixedopen-gripper shadow-envelopecheck retainingthresholds','recordall54assignedintents,missing/unknown/failures; reserve72of900controls','same-timecamera/truthintegrityonly; no cross-rollout equalityqualification','freezeimplementation/source/config/layoutmanifesthashes beforefirstphysicsstep'],
 'original_guide_complete':False,'model_acceptance_or_controller_promotion':False
 }
 save(E/'configs/protocol.json',protocol);save(R/'PROTOCOL.json',protocol)
 evidence={
 'src/wm_symbolic/physical_oracle.py':'step55,hold70;run130 containsoutcome/refusal/retry andmustnotbecalled',
 'src/wm_symbolic/physical_oracle_v2.py':'shadowcheck39,monitor64,plan_lift83; unchanged thresholds, notpromoted',
 'src/wm_symbolic/env_adapter.py':'model_view52 hasdepth;segmentation62 addsprivilegedfields; explicitnarrowallowlistrequired',
 'vendor/calvin/calvin_env/calvin_env/robot/robot.py':'get_observation181 Euler15;apply_action244 andcontrol_gripper359 nativecommands',
 'scripts/evaluate_physical_oracle_v2.py':'physics_guard blocksworldmutation butdoesnotalreadylistrestoreState/loadBullet; newwrappermustblockthem',
 'experiments/relation_GT_boxes36_ABC_20261018/source/collect.py':'MotorAdapter23 andcapture65 reusable; run()/postcleanupnotreused',
 'experiments/holding_action_fork_feasibility12_v3_20261018/source/collect.py':'capture_native143 andtarget/supportaudit158 read-only reference; restorepathclosed',
 'reports/holding_static_context_diagnostic_20261018/REPORT.md':'supportnegativecoveragegap andstaticoptimizationclosed',
 'reports/holding_matched_input_diagnostic_20261018/REPORT.md':'matchedRGBpredictiveevidence andsameboxnot sameRGB',
 'reports/relation_GT_boxes36_ABC_20261018/REPORT.md':'36roots264states andoracleboxeslimitations',
 'reports/native_restore_semantics_root1_final_20261018/supported_start_geometry.json':'20close alreadyHolding;0.765mmtableclearance, cannotassumepresupport',
 'reports/acquisition_scene_surface_pilot8_20261018/REPORT.md':'historicalpostprocessingnon-targetcontacts; omittranslate/returncleanup',
 'runs/physical_oracle_v2/CONTROLLER_NOT_PROMOTED.json':'27chainspassedbut134/135feasible; noexecutorpromotion',
 'reports/holding_action_fork_feasibility12_v3_20261018/AUDIT.json':'exactforknegativequalificationclosed',
 'source/世界模型与TS-LTL符号规划_核心预实验方案.md':'authoritativeguide, noacceptance/scalesubstitution'}
 refs=[{'path':str(P/p),'sha256':sha(P/p),'relevance':v}for p,v in evidence.items()];save(E/'configs/source_evidence.json',refs);shutil.copy2(E/'configs/source_evidence.json',R/'source_evidence.json')
 prior=['relation_GT_boxes_holding32_20261018','acquisition_scene_surface_pilot8_20261018','relation_GT_boxes36_ABC_20261018','holding_matched_input_diagnostic_20261018','holding_static_context_diagnostic_20261018','holding_action_fork_feasibility12_20261018','native_restore_semantics_root1_20261018','native_restore_semantics_root1_fix01_20261018','native_restore_semantics_root1_final_20261018','holding_action_fork_feasibility12_v2_20261018','holding_action_fork_feasibility12_v3_20261018'];oldfiles={}
 for n in prior:
  f=P/'experiments'/n/'configs/delivery_manifest.json';oldfiles.update(read(f)['sha256']);oldfiles[str(f)]=sha(f)
 for p,h in oldfiles.items():assert sha(p)==h,p
 verification={'utc':protocol['utc'],'design_only':True,'planned_slots':54,'observed_trajectories':0,'new_physics_steps':0,'new_render_calls':0,'new_model_fits':0,'all_allocation_marginals_and_full_color_geometry_start_action_crossing_verified':True,'train_roots':36,'dev_roots':18,'each_scene_start_action_cell':2,'registered_rootIDs_and_seeds_disjoint':True,'actual_layout_uniqueness_still_requires_precollection_check':True,'no_success_or_label_read_by_assignment_generator':True,'historical_files_hash_verified':len(oldfiles),'source_evidence_files':len(refs),'no_new_implementation_or_runtime_validation_claim':True}
 save(E/'results/DESIGN_ONLY.json',verification);save(R/'VERIFICATION.json',verification)
 shutil.copy2(Path(__file__),E/'source/build_protocol.py');shutil.copy2(Path(__file__).parent/'PROTOCOL.md',R/'PROTOCOL.md')
 print(json.dumps({'experiment':str(E),'report':str(R/'PROTOCOL.md'),'protocol_sha256':sha(E/'configs/protocol.json'),'assignments_sha256':sha(E/'configs/assignments.csv'),'verification':verification},ensure_ascii=False))

if __name__=='__main__':main()
