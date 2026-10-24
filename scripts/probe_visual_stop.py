"""Paired versioned executor trial; GT is logged/scored but cannot choose stopping."""
from pathlib import Path
from collections import deque
import json,time,random,hashlib,datetime
import numpy as np,torch,hydra,imageio.v2 as imageio
from omegaconf import OmegaConf
from wm_symbolic.env_adapter import CalvinAdapter,save_npz
from wm_symbolic.skill_runner import FrozenHulc
from wm_symbolic.visual_stop import VisualStopper
from wm_symbolic.checkpointing import PauseRequest
P=Path('/home/robot/research/projects/wm_symbolic');out=P/'runs/executor_visual_stop_v2_paired';out.mkdir(exist_ok=False);config_path=P/'configs/executor_visual_stop_v2.json';cfg=json.loads(config_path.read_text());config_sha=hashlib.sha256(config_path.read_bytes()).hexdigest();pause=PauseRequest();torch.set_num_threads(4)
policy=FrozenHulc(P/'models/HULC_ABC_D',P/'models/HULC_ABC_D/saved_models/HULC_ABC_D.ckpt',P/'data/raw/calvin_debug_dataset/validation/lang_paraphrase-MiniLM-L3-v2/embeddings.npy');commands=OmegaConf.load(P/'vendor/calvin/calvin_models/conf/annotations/new_playtable_validation.yaml');checker=hydra.utils.instantiate(OmegaConf.load(P/'vendor/calvin/calvin_models/conf/callbacks/rollout/tasks/new_playtable_tasks.yaml'))
budgets=json.loads((P/'configs/executor_budgets_pilot.json').read_text())['steps'];record={'status':'running','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'controller_weights_sha256':policy.identity['checkpoint_sha256'],'visual_config_sha256':config_sha,'calibration_frozen_before_trials':True,'GT_used_for_stopping':False,'original_active_manifest_changed':False,'root_count':6,'skill_names':['open_drawer','close_drawer','lift_red_block_table','lift_blue_block_table','lift_pink_block_table'],'trials':[],'pairs':[]}
def serial(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,bytes):return x.decode(errors='replace')
 raise TypeError(type(x).__name__)
def save():
 tmp=out/'results.tmp';tmp.write_text(json.dumps(record,ensure_ascii=False,indent=2));tmp.replace(out/'results.json')
def digest(obs):return hashlib.sha256(b''.join(obs['rgb_obs'][k].tobytes() for k in sorted(obs['rgb_obs']))+obs['robot_obs'].tobytes()).hexdigest()
save()
for root_index in range(6):
 seed=root_index if root_index<3 else 6100+root_index
 for skill in record['skill_names']:
  paired=[]
  for candidate in [False,True]:
   version='v2_visual_stop' if candidate else 'v1_fixed_budget';env=CalvinAdapter(P,'C',seed,tabletop=True,segmentation=True)
   try:
    if root_index<3:
     source=P/f'runs/stage0_colour_binding_20261013_151842/C_root{root_index:03d}_evaluator_only.npz'
     with np.load(source) as z:root={'environment':'C','robot_obs':z['robot_obs'],'scene_obs':z['scene_obs']}
     env.reset(root)
    else:env.reset()
    if skill in ['open_drawer','close_drawer']:
     for door in env.env.scene.doors:
      if door.name=='base__drawer':door.reset(.22 if skill=='close_drawer' else 0.)
    for _ in range(4):env.step(np.array([0,0,0,0,0,0,1.],np.float32))
    initial=env.observe();root_hash=digest(initial);private_start=env.evaluator_snapshot();torch.manual_seed(7400+root_index);np.random.seed(7400+root_index);random.seed(7400+root_index);policy.reset();detector=VisualStopper(skill,initial,cfg)
    folder=out/f'C_root{root_index:03d}_{skill}_{version}';folder.mkdir();save_npz(folder/'before.npz',initial);writers={key:imageio.get_writer(str(folder/f'{key}.mp4'),fps=30,codec='libx264',macro_block_size=1,quality=7) for key in initial['rgb_obs']};controls=[];robots=[];scenes=[];infos=[private_start['info']];decisions=[];reason='fixed_budget_exhausted';budget=budgets['drawer' if 'drawer' in skill and not skill.startswith('lift_') else 'lift'];begin=time.monotonic();obs=initial
    try:
     for step in range(1,budget+1):
      action=policy.step(obs,commands[skill][0]);obs=env.step(action);snapshot=env.evaluator_snapshot();controls.append(action.copy());robots.append(obs['robot_obs']);scenes.append(snapshot['scene_obs']);infos.append(snapshot['info'])
      for key,writer in writers.items():writer.append_data(obs['rgb_obs'][key])
      evidence=detector.check(obs,step)
      if evidence:
       decisions.append(evidence)
       if evidence['decision'] and candidate:reason=evidence['decision'];break
      if pause.requested:reason='paused_by_user';break
    finally:
     for writer in writers.values():writer.close()
     np.savez_compressed(folder/'controls_and_evaluator_state.npz',actions=controls,robot_obs=robots,scene_obs=scenes);save_npz(folder/'after.npz',obs)
     for i,frame in enumerate(env.history):save_npz(folder/f'boundary_history_{i}.npz',frame)
     with (folder/'trajectory_evaluator_only.jsonl').open('w') as f:
      for i,info in enumerate(infos):f.write(json.dumps({'step':i,'info':info},default=serial)+'\n')
     (folder/'visual_decisions.json').write_text(json.dumps(decisions,indent=2))
    hits=[i for i in range(1,len(infos)) if skill in checker.get_task_info_for_set(infos[0],infos[i],{skill})];actual=bool(skill in checker.get_task_info_for_set(infos[0],infos[-1],{skill}))
    objects=infos[0]['scene_info']['movable_objects'];changes={name:{'max_height_rise_m':max(info['scene_info']['movable_objects'][name]['current_pos'][2] for info in infos)-value['current_pos'][2],'positive_robot_contact_steps':sum(any(c[2]==value['uid'] and c[9]>0 for c in info['robot_info']['contacts']) for info in infos[1:])} for name,value in objects.items()}
    summary={'root_index':root_index,'root_seed':seed,'skill':skill,'command':commands[skill][0],'executor_version':version,'public_root_sha256':root_hash,'requested_steps':budget,'actual_steps':len(controls),'stop_reason':reason,'visual_claimed_completion':reason=='visual_completed','official_success_at_end':actual,'official_first_success_step':hits[0] if hits else None,'official_ever_successful':bool(hits),'actual_object_evidence_evaluator_only':changes,'wall_seconds':time.monotonic()-begin,'simulated_seconds':len(controls)/30,'GT_used_for_stopping':False,'visual_config_sha256':config_sha if candidate else None,'checkpoint_sha256':policy.identity['checkpoint_sha256'],'run':str(folder)}
    (folder/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));record['trials'].append(summary);paired.append(summary);save();print('VISUAL_STOP_TRIAL',root_index,skill,version,len(controls),reason,actual,flush=True)
   finally:env.close()
   if pause.requested:record['status']='paused';save();raise SystemExit(0)
  with np.load(Path(paired[0]['run'])/'controls_and_evaluator_state.npz') as z:first=z['actions']
  with np.load(Path(paired[1]['run'])/'controls_and_evaluator_state.npz') as z:second=z['actions']
  prefix_equal=np.array_equal(first[:len(second)],second);root_equal=paired[0]['public_root_sha256']==paired[1]['public_root_sha256'];assert root_equal and prefix_equal,'Paired comparison changed root or action prefix'
  record['pairs'].append({'root_index':root_index,'skill':skill,'same_root':root_equal,'same_action_prefix_until_stop':prefix_equal,'v1_final_success':paired[0]['official_success_at_end'],'v2_final_success':paired[1]['official_success_at_end'],'v1_controls':paired[0]['actual_steps'],'v2_controls':paired[1]['actual_steps'],'v2_stop_reason':paired[1]['stop_reason']});save()
record.update(status='complete',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),user_review_required=True,not_promoted_to_formal_collection=True);save();print('VISUAL_STOP_PAIRED_COMPLETE',len(record['pairs']),flush=True)
