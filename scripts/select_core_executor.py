"""Retrospective, user-authorized scoring amendment; no simulator execution."""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
from omegaconf import OmegaConf
from wm_symbolic.calibrated_labels import calibrated_labels_at
from wm_symbolic.state_labels import SPEC

P=Path('/home/robot/research/projects/wm_symbolic');V=P/'runs/physical_oracle_v1';OUT=P/'runs/core_executor_selection_20261014'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
def label(frame,cfg):
    ids={'block_blue':2,'block_red':3,'block_pink':4};contacts=frame['contacts'];robot=frame['robot_obs'];state=frame['scene_obs']
    info={'robot_info':{'uid':0,'tcp_pos':robot[:3],'gripper_action':robot[-1]},'scene_info':{'fixed_objects':{'table':{'uid':5,'links':{'base_link':-1,'drawer_link':3,'plank_link':6}}},'movable_objects':{name:{'uid':uid,'current_pos':frame['objects'][name][:3],'contacts':[c for c in contacts if c[1]==uid or c[2]==uid]} for name,uid in ids.items()},'doors':{'base__slide':{'current_state':state[0]},'base__drawer':{'current_state':state[1]}},'lights':{'lightbulb':{'logical_state':state[4]},'led':{'logical_state':state[5]}}}}
    return calibrated_labels_at(info,cfg)
def main():
 OUT.mkdir(exist_ok=False);now=datetime.datetime.now(datetime.timezone.utc).isoformat()
 before_files=[V/'summary.json',V/'controller_freeze.json',P/'reports/physical_oracle_v1.html',P/'src/wm_symbolic/physical_oracle.py',P/'data/data_manifest.json']
 hashes={str(f.relative_to(P)):sha(f) for f in before_files}
 cfg=OmegaConf.to_container(OmegaConf.load(P/'vendor/calvin/calvin_env/conf/scene/calvin_scene_C.yaml'),resolve=False)
 rows=[dict(load(f),result_path=str(f.relative_to(P))) for f in sorted((V/'independent_acceptance').glob('*/*/*/result.json'))]
 exceptions=[]
 for r in rows:
  if r['conservative_pass']:continue
  assert r['root']=='C_acceptance_02' and r['skill']=='lift_pink_block_table' and r['unrequested_device_state_change']
  assert r['status']=='success' and r['official_success'] and r['held_private']==['block_pink']
  tracepath=P/r['result_path'];trace=load(tracepath.parent/'trace_private.json');labels=[label(x,cfg) for x in trace]
  non_target_indices=[0,1,3,4] # red, blue, drawer, sliding door; pink is requested target.
  invariants={'non_target_object_and_mechanism_categories_unchanged':all(all(y['categorical'][i]==labels[0]['categorical'][i] for i in non_target_indices) for y in labels),
   'non_target_category_validity_all_true':all(all(y['categorical_valid'][i] for i in non_target_indices) for y in labels),
   'light_bits_unchanged':all(y['binary']==labels[0]['binary'] for y in labels),
   'only_named_pink_held_at_end':labels[-1]['holding_derived']==[False,False,True],
   'physics_guard_passed':r['physics_guard_passed'],
   'non_target_drift_under_original10mm':r['max_unrequested_object_displacement_m']<.01,
   'contact_penetration_under_original10mm':r['max_robot_contact_penetration_m']<.01}
  assert all(invariants.values()),invariants
  scene=np.asarray([x['scene_obs'] for x in trace]);slide=scene[:,0];delta=float(np.max(np.abs(slide-slide[0])))
  assert delta<.01 # Evidence verification for these two recorded cases, not a new blanket tolerance.
  exceptions.append({'result_path':r['result_path'],'case':r['case'],'before_strict_pass':False,'amended_task_pass':True,'amendment_is_post_hoc':True,'user_reason':'误碰的程度很小可以忽略不计，认为是成功','slider_max_change_mm':delta*1000,'slider_categorical_state_entire_trace':'right','slider_right_threshold_m':SPEC['slider_right_max'],'non_target_max_displacement_mm':r['max_unrequested_object_displacement_m']*1000,'before_categorical':labels[0]['categorical'],'after_categorical':labels[-1]['categorical'],'before_binary':labels[0]['binary'],'after_binary':labels[-1]['binary'],'invariants':invariants,'raw_trace_sha256':sha(tracepath.parent/'trace_private.json')})
 assert len(exceptions)==2
 excpaths={x['result_path'] for x in exceptions}
 amended=[dict(result_path=r['result_path'],expected=r['expected'],root=r['root'],case=r['case'],skill=r['skill'],strict_pass=r['conservative_pass'],task_level_pass=bool(r['conservative_pass'] or r['result_path'] in excpaths),post_hoc_exception=r['result_path'] in excpaths) for r in rows]
 pos=[r for r in amended if r['expected']=='success'];neg=[r for r in amended if r['expected']!='success'];chainroots=sorted({r['root'] for r in amended if r['case']=='chain'})
 chains=sum(all(r['task_level_pass'] for r in amended if r['root']==root and r['case']=='chain') for root in chainroots)
 protocol={'id':'core_task_success_user_amendment_20261014','recorded_utc':now,'user_instruction':'误碰的程度很小可以忽略不计，认为是成功','instruction_source':'Explicit latest human instruction relayed by parent thread01a11850-0ce1-73fb-9aa2-4e19130a1031 on2026-10-14',
 'post_hoc':True,'meaning':'Task-level acceptance of the recorded small sliding-door side effect. Original strict17/18 score is preserved; revised18/18 is retrospective, not predeclared.',
 'task_success_requirements':['Current named skill reaches official and fixed-controller stable goal.','Correct named object and intended ownership/support/holding relation.','No change to task-relevant non-target logical predicates or object ownership.','No violation of explicitly specified task forbidden conditions.','Physical execution only; no teleport/collision bypass/automatic prerequisite action.'],
 'incidental_motion_scope':'The two recorded manifestations of the same reviewed C_acceptance_02 pink-grasp side effect are accepted here. No new blanket numeric collision tolerance is inferred for arbitrary future episodes. New cases keep raw continuous values and strict score alongside task outcome; unreviewed deviations are flagged, never silently excused.',
 'constraints_limit':'These controller core tests did not specify a full future LTL safety formula; no claim that arbitrary future forbidden conditions were checked. Any future formula prohibition remains binding, even if a motion looks small.',
 'training_boundary':'Raw motion labels and images are not changed. Task-success amendment is scoring metadata only. GT, success bits and refusal reasons remain private, not WM/high-level inputs.',
 'exceptions':exceptions}
 (OUT/'scoring_protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2));(OUT/'amended_scores.json').write_text(json.dumps(amended,indent=2))
 selection={'selected_utc':now,'selected_version':'physical_oracle_v1_candidate3','python_class':'wm_symbolic.physical_oracle.PhysicalOracle','public_entrypoint':'execute','controller_sha256':sha(P/'src/wm_symbolic/physical_oracle.py'),'frozen_source':'source/controller_snapshots/physical_oracle_v1/physical_oracle.py',
 'selection_reason':'v1 has no observed goal execution failure in90 feasible held-out calls; its only2 strict failures were the same user-accepted small slider side effect. v2 introduced a genuine planner candidate-search failure in1/135 feasible calls; no reason to require its stricter avoidance for the present task.',
 'supported_skills':['open_drawer','close_drawer','lift_red_block_table','lift_blue_block_table','lift_pink_block_table','place_in_drawer'],
 'limits':{'max_controls':900,'max_attempts':2,'stable_goal_controls':12},
 'observed_scope':'Native A/B/C separated tabletop blocks, tested colors/yaws and four-step drawer storage chain. Nine development and18 held-out v1 roots. No D, clutter/stacking, arbitrary shelf reach or remaining19 skills verified.',
 'original_strict':{'chains':[17,18],'feasible':[88,90],'correct_refusals':[90,90]},'user_amended_retrospective':{'chains':[chains,len(chainroots)],'feasible':[sum(r['task_level_pass'] for r in pos),len(pos)],'correct_refusals':[sum(r['task_level_pass'] for r in neg),len(neg)]},
 'scoring_protocol':'runs/core_executor_selection_20261014/scoring_protocol.json','controller_frozen_for_six_skill_WM_core_validation':True,'general_reliability_guarantee':False,'original_full_guide_completed':False,
 'formal_collection_authorized':False,'formal_training_authorized':False,'formal_pipeline_resumed':False,'new1200_proposal':'runs/physical_oracle_v2/wm_1200_proposal_review.json','next':'Return research focus to the six-skill world-model dataset design; preserve60/20/20 proposal as unexecuted until separately authorized.'}
 assert selection['user_amended_retrospective']=={'chains':[18,18],'feasible':[90,90],'correct_refusals':[90,90]}
 (OUT/'selection.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2))
 dest=P/'configs/controllers';dest.mkdir(exist_ok=True);(dest/'selected_core_executor.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2))
 stop={'stopped_utc':now,'reason':'User accepts the reviewed small side effect; stop further controller optimization and do not expand skills. Select frozenv1 for six-skill core work.', 'v3_last_probe':'Completed at safe episode boundary before stop; closed-drawer red grasp succeeded with checked front-clearance route, confirmingv2 failure was search/path insufficiency, not demonstrated physical unreachability.', 'v3_full_regression_run':False,'v3_new_acceptance_run':False,'all_v1_v2_v3_evidence_preserved':True}
 (P/'runs/physical_oracle_v3/STOPPED_BY_USER.json').write_text(json.dumps(stop,indent=2))
 for f in before_files:assert sha(f)==hashes[str(f.relative_to(P))]
 (OUT/'preservation_audit.json').write_text(json.dumps({'checked_utc':now,'original_files_unchanged':hashes,'main_transitions':load(P/'data/data_manifest.json')['real_transitions'],'collection_hold':(P/'runs/PAUSED_BY_USER.json').exists(),'training_hold':(P/'runs/FORMAL_TRAINING_HOLD.json').exists()},indent=2))
 for f in OUT.glob('*.json'):f.chmod(0o444)
 h=load(P/'runs/continuation_handoff.json');h.update(stage='core_executor_v1_selected_with_user_posthoc_scoring',active_pipeline=None,active_session=None,selected_core_executor='configs/controllers/selected_core_executor.json',latest_controller_result='runs/core_executor_selection_20261014/selection.json',latest_scoring_protocol='runs/core_executor_selection_20261014/scoring_protocol.json',controller_optimization_stopped_by_user=True,core_result_original_strict='17/18 chains;88/90 feasible;90/90 correct refusals',core_result_user_amended_posthoc='18/18 chains;90/90 feasible;90/90 correct refusals',next='World-model dataset design for6-skill reduced scope. Do not run proposed1200 collection/training without authorization.',updated_utc=now);(P/'runs/continuation_handoff.json').write_text(json.dumps(h,indent=2))
 print(json.dumps({'selection':selection,'exceptions':exceptions},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
