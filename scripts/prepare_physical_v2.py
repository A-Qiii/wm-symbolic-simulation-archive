from pathlib import Path
import json,hashlib,datetime,shutil
import numpy as np
import pybullet as p
from PIL import Image
from wm_symbolic.env_adapter import CalvinAdapter,save_npz
P=Path('/home/robot/research/projects/wm_symbolic');B=P/'runs/physical_oracle_v2';OLD=P/'runs/physical_oracle_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 B.mkdir(exist_ok=False)
 protocol={'version':'physical_oracle_v2','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'authorization':'Repair full-hand/gripper obstacle clearance; preserve v1, old18 acceptance roots are now regression only; fresh roots frozen before development. No new WM collection/training.',
 'development_and_regression':'v1 development9 + old acceptance18; v1 outcome remains17/18 and immutable.',
 'fresh_acceptance':'27 new nativeABC tabletop roots:9 per environment,3 per named color, seeds91000/91100/91200+i. Same spacing/yaw distribution asv1, no outcome screening; no D.',
 'criteria_unchanged':json.loads((OLD/'controller_freeze.json').read_text())['acceptance_plan'],
 'runtime_limits':{'max_controls_per_skill':900,'max_attempts':2,'stable_completion_controls':12},
 'planned_method':'Private separate-client kinematic collision queries over exact robot/table/block collision geometry. Only actual robot motor actions change execution world; no actual object reset, direct table actuation, collision disable or external force. GT never supplied to WM/planner.',
 'main_manifest_sha256':sha(P/'data/data_manifest.json'),'v1_freeze_sha256':sha(OLD/'controller_freeze.json')}
 protocol['criteria_unchanged']['roots']=27;protocol['criteria_unchanged']['chain_episodes']=27;protocol['criteria_unchanged']['negative_episodes']=27
 (B/'protocol.json').write_text(json.dumps(protocol,indent=2))
 rows=[]
 for ei,e in enumerate('ABC'):
  env=CalvinAdapter(P,e,91000+ei*100,tabletop=True)
  try:
   lo,hi=np.asarray(env.env.scene.surfaces['table'])
   for i in range(9):
    seed=91000+100*ei+i;rng=np.random.RandomState(seed);names=rng.permutation(['block_red','block_blue','block_pink'])
    layout={name:[float(lo[0]+(hi[0]-lo[0])*frac+rng.uniform(-.008,.008)),float(rng.uniform(-.13,-.065)),.46] for name,frac in zip(names,[.18,.5,.82])}
    for obj in env.env.scene.movable_objects:obj.initial_pos=np.array(layout[obj.name]);obj.initial_orn=np.array(p.getQuaternionFromEuler([0,0,rng.uniform(-np.pi,np.pi)]))
    obs=env.reset()
    for _ in range(24):obs=env.step([obs['robot_obs'][:3],obs['robot_obs'][3:6],1])
    root=env.evaluator_snapshot();rid=f'{e}_acceptance_v2_{i:02d}';d=B/'roots'/rid;d.mkdir(parents=True)
    np.savez_compressed(d/'reset_evaluator_only.npz',robot_obs=root['robot_obs'],scene_obs=root['scene_obs']);save_npz(d/'initial_public.npz',obs)
    for camera,pix in obs['rgb_obs'].items():Image.fromarray(pix).save(d/f'{camera}.png')
    rows.append({'root_id':rid,'split':'acceptance','environment':e,'seed':seed,'chain_color':['red','blue','pink'][i%3],'reset_sha256':sha(d/'reset_evaluator_only.npz')})
  finally:env.close()
 oldrows=json.loads((OLD/'roots_manifest.json').read_text());assert not ({r['reset_sha256'] for r in rows}&{r['reset_sha256'] for r in oldrows})
 (B/'roots_manifest.json').write_text(json.dumps(rows,indent=2))
 for f in [B/'protocol.json',B/'roots_manifest.json',*B.glob('roots/*/reset_evaluator_only.npz')]:f.chmod(0o444)
 shutil.copyfile('/home/robot/Documents/Codex/2026-10-13/task-4/library_deliverables/library_upload_receipts.json',B/'v1_video_library_receipts.json')
 notes={'status':'proposal_only_not_execution_authorization','proposed_total':1200,'roots':100,'split_roots':[60,20,20],'split_transitions':[720,240,240],'executor_scope':6,'compatibility':'Same-root branches must share split; label actual physical changes, including identity after refusal. Controller refusal reason/status/GT not model or planner input.',
 'specific_risk':'With six skills, grasping while drawer closed creates a held-object state with no supported put-back and no legal empty-gripper drawer open. A naive wrong-order branch can become3 successive no-op refusals.',
 'minimal_change':'Keep 3x4 if desired; use one successful chain plus2 counterfactual branches with at most one precondition violation each and remaining feasible named actions. Example1: empty place,open,lift,place. Example2:open,lift,close-while-held(refuse),place. Balance colors across branches. Do not silently insert prerequisites.',
 'additional_checks':['Before collecting100 roots, simulate the symbolic schedule offline for refusal fraction; do not assert success from schedule.','Use separate new WM roots, not controller development/regression roots.','Predetermine composition holdout in addition to root split; random seeds alone do not test new task combinations.','Report nominal1200 and actual counts separately if interrupted; no replacing failures.','Mark this as reduced6-skill core validation, not original6000-transition guide.']}
 (B/'wm_1200_proposal_review.json').write_text(json.dumps(notes,indent=2))
 h=json.loads((P/'runs/continuation_handoff.json').read_text());h.update(stage='physical_oracle_v2_development',active_pipeline='collision-aware physical controller development',latest_controller_protocol='runs/physical_oracle_v2/protocol.json',v1_library_video_ids=['libfile_012a0cbbf8c081918501dc014fc578d9','libfile_b66d5fd6c2ec8191a592e534b84625da'],fresh_holdout_roots=27,wm_dataset_proposal='runs/physical_oracle_v2/wm_1200_proposal_review.json',next='Develop full-hand collision-aware grasp path on old roots; freeze before27 fresh acceptance plus18 old-root regression.',updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());(P/'runs/continuation_handoff.json').write_text(json.dumps(h,indent=2))
 print(json.dumps({'frozen_new_roots':27,'old_v1_unchanged':True,'library_ids':h['v1_library_video_ids']}))
if __name__=='__main__':main()
