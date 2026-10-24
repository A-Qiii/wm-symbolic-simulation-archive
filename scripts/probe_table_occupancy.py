"""Bounded diagnostic only: move one non-target block; never change main roots."""
from pathlib import Path
import json, random, hashlib, datetime
import numpy as np
import torch, hydra, imageio.v2 as imageio
from omegaconf import OmegaConf
from wm_symbolic.env_adapter import CalvinAdapter, save_npz
from wm_symbolic.skill_runner import FrozenHulc
from wm_symbolic.visual_stop import VisualStopper
from wm_symbolic.checkpointing import PauseRequest

P=Path('/home/robot/research/projects/wm_symbolic')
out=P/'runs/executor_table_occupancy_diagnostic_v1'
out.mkdir(exist_ok=False)
pause=PauseRequest(); torch.set_num_threads(4)
manifest=P/'data/data_manifest.json'
manifest_sha=hashlib.sha256(manifest.read_bytes()).hexdigest()
colors=['red','blue','pink']
source=json.loads((P/'runs/executor_visual_stop_v2_paired/results.json').read_text())
plan={'status':'running','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'hypothesis':'A third tabletop object contributes to color acquisition failure.',
 'scope':'C diagnostic only. Does not replace task distribution or enter training/calibration/test data.',
 'fixed':['HULC weights','command','policy RNG per root','target initial pose','robot initial pose','camera','360 control budget'],
 'intervention':'Move each non-target block separately to midpoint of native C slider_left surface, keeping its orientation. No hidden state enters policy or stopping.',
 'conditions':['three_on_table','non_target_1_to_slider','non_target_2_to_slider'],
 'roots':6,'colors':colors,'planned_trials':54,'main_manifest_sha256_before':manifest_sha,
 'no_early_stopping':True,'GT_used_for_policy_or_stopping':False,'trials':[]}
def serial(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,bytes):return x.decode(errors='replace')
 raise TypeError(type(x).__name__)
def save():
 tmp=out/'results.tmp';tmp.write_text(json.dumps(plan,ensure_ascii=False,indent=2,default=serial));tmp.replace(out/'results.json')
def digest(obs):
 return hashlib.sha256(b''.join(obs['rgb_obs'][k].tobytes() for k in sorted(obs['rgb_obs']))+obs['robot_obs'].tobytes()).hexdigest()
save()
policy=FrozenHulc(P/'models/HULC_ABC_D',P/'models/HULC_ABC_D/saved_models/HULC_ABC_D.ckpt',P/'data/raw/calvin_debug_dataset/validation/lang_paraphrase-MiniLM-L3-v2/embeddings.npy')
plan['controller_identity']=policy.identity
commands=OmegaConf.load(P/'vendor/calvin/calvin_models/conf/annotations/new_playtable_validation.yaml')
checker=hydra.utils.instantiate(OmegaConf.load(P/'vendor/calvin/calvin_models/conf/callbacks/rollout/tasks/new_playtable_tasks.yaml'))
cfg=json.loads((P/'configs/executor_visual_stop_v2.json').read_text())
for root_index in range(6):
 seed=root_index if root_index<3 else 6100+root_index
 for color in colors:
  skill=f'lift_{color}_block_table'
  original=next(x for x in source['trials'] if x['root_index']==root_index and x['skill']==skill and x['executor_version']=='v1_fixed_budget')
  conditions=[None]+[c for c in colors if c!=color]
  reference_private=None
  for moved in conditions:
   condition='three_on_table' if moved is None else f'{moved}_to_slider'
   folder=out/f'C_root{root_index:03d}_{skill}_{condition}';folder.mkdir()
   env=CalvinAdapter(P,'C',seed,tabletop=True,segmentation=True)
   try:
    if root_index<3:
     with np.load(P/f'runs/stage0_colour_binding_20261013_151842/C_root{root_index:03d}_evaluator_only.npz') as z:
      env.reset({'environment':'C','robot_obs':z['robot_obs'],'scene_obs':z['scene_obs']})
    else:env.reset()
    if moved:
     obj=next(x for x in env.env.scene.movable_objects if x.name=='block_'+moved)
     state=obj.get_state().copy();state[:3]=np.mean(np.asarray(env.config.scene.surfaces.slider_left),axis=0);obj.reset(state)
    for _ in range(4):env.step(np.array([0,0,0,0,0,0,1.],np.float32))
    obs=env.observe(); initial_hash=digest(obs);initial=env.evaluator_snapshot();save_npz(folder/'before.npz',obs)
    target=initial['info']['scene_info']['movable_objects']['block_'+color]
    if moved is None:
     assert digest(obs)==original['public_root_sha256'],'Original root recreation changed'
     reference_private=initial
    unaffected=['block_'+c for c in colors if c!=moved]
    pose_differences={name:float(np.linalg.norm(np.array(initial['info']['scene_info']['movable_objects'][name]['current_pos'])-reference_private['info']['scene_info']['movable_objects'][name]['current_pos'])) for name in unaffected}
    robot_difference=float(np.max(np.abs(obs['robot_obs']-reference_private['robot_obs'])))
    # Differences after settling are recorded, not silently screened away.
    detector=VisualStopper(skill,obs,cfg);shadow=[];first_decision=None
    torch.manual_seed(7400+root_index);np.random.seed(7400+root_index);random.seed(7400+root_index);policy.reset()
    controls=[];robots=[];scenes=[];infos=[initial['info']]
    writers={k:imageio.get_writer(str(folder/f'{k}.mp4'),fps=30,codec='libx264',macro_block_size=1,quality=7) for k in obs['rgb_obs']}
    try:
     for step in range(1,361):
      action=policy.step(obs,commands[skill][0]);obs=env.step(action);private=env.evaluator_snapshot()
      controls.append(action.copy());robots.append(obs['robot_obs']);scenes.append(private['scene_obs']);infos.append(private['info'])
      for k,w in writers.items():w.append_data(obs['rgb_obs'][k])
      decision=detector.check(obs,step)
      if decision:
       shadow.append(decision)
       if decision['decision'] and first_decision is None:first_decision=decision
      if pause.requested:break
    finally:
     for w in writers.values():w.close()
     save_npz(folder/'after.npz',obs)
     np.savez_compressed(folder/'controls_and_evaluator_state.npz',actions=controls,robot_obs=robots,scene_obs=scenes)
     (folder/'initial_evaluator_only.json').write_text(json.dumps(initial,default=serial))
     (folder/'visual_shadow_decisions.json').write_text(json.dumps(shadow,default=serial))
     with (folder/'trajectory_evaluator_only.jsonl').open('w') as f:
      for i,info in enumerate(infos):f.write(json.dumps({'step':i,'info':info},default=serial)+'\n')
    hits=[i for i in range(1,len(infos)) if skill in checker.get_task_info_for_set(infos[0],infos[i],{skill})]
    acquisition=None
    for step,info in enumerate(infos[1:],1):
     for actual in colors:
      name='block_'+actual;obj=info['scene_info']['movable_objects'][name];rise=obj['current_pos'][2]-infos[0]['scene_info']['movable_objects'][name]['current_pos'][2]
      if robots[step-1][-1]<0:
       if rise>.05 and any(c[2]==obj['uid'] and c[9]>0 for c in info['robot_info']['contacts']):
        acquisition={'step':step,'color':actual,'rise_m':rise};break
     if acquisition:break
    rec={'root_index':root_index,'skill':skill,'moved_non_target':moved,'condition':condition,'controls':len(controls),'run':str(folder),'initial_public_sha256':initial_hash,
      'unchanged_object_position_difference_after_settle_m':pose_differences,'robot_max_difference_after_settle':robot_difference,'ever_success':bool(hits),'first_success_step':hits[0] if hits else None,'final_success':len(controls) in hits,'first_contacted_and_raised':acquisition,'first_public_visual_decision':first_decision}
    if moved is None:
     with np.load(Path(original['run'])/'controls_and_evaluator_state.npz') as z:rec['exact_original_actions_reproduced']=np.array_equal(z['actions'],np.asarray(controls))
     assert rec['exact_original_actions_reproduced'] or pause.requested,'Original policy trajectory changed'
    if first_decision:
     step=first_decision['control_step'];rec['offline_success_at_shadow_stop']=step in hits
    (folder/'summary.json').write_text(json.dumps(rec,default=serial,indent=2));plan['trials'].append(rec);save()
    print('OCCUPANCY',root_index,color,condition,len(controls),bool(hits),rec['final_success'],acquisition,flush=True)
   finally:env.close()
   if pause.requested:plan.update(status='paused',reason=pause.reason);save();raise SystemExit(0)
plan.update(status='complete',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),main_manifest_sha256_after=hashlib.sha256(manifest.read_bytes()).hexdigest())
assert plan['main_manifest_sha256_after']==manifest_sha
save();print('OCCUPANCY_COMPLETE',len(plan['trials']),flush=True)
