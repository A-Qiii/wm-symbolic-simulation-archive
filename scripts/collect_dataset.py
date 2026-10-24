"""Resumable, root-grouped real collection. Whole six-skill episodes only."""
from pathlib import Path
import argparse,json,time,random,hashlib,os
import numpy as np,torch,hydra
from omegaconf import OmegaConf
from wm_symbolic.env_adapter import CalvinAdapter,save_npz
from wm_symbolic.skill_runner import FrozenHulc,execute_fixed
from wm_symbolic.checkpointing import PauseRequest
from wm_symbolic.root_registry import active_root_directory
import datetime
P=Path('/home/robot/research/projects/wm_symbolic')
if (P/'runs/PAUSED_BY_USER.json').exists():raise SystemExit('Paused by user')
pause=PauseRequest()
args=argparse.ArgumentParser();args.add_argument('--split',choices=['train','dev','calibration','test'],default='train');args.add_argument('--roots',type=int,required=True);args.add_argument('--branches',type=int,default=2);args.add_argument('--stop-after',type=int,default=0);a=args.parse_args()
catalog=json.loads((P/'configs/skills.json').read_text());ids=[s['skill'] for s in catalog];family={s['skill']:s['family'] for s in catalog}
budgets=json.loads((P/'configs/executor_budgets_pilot.json').read_text())['steps']
ann=OmegaConf.load(P/'vendor/calvin/calvin_models/conf/annotations/new_playtable_validation.yaml')
checker=hydra.utils.instantiate(OmegaConf.load(P/'vendor/calvin/calvin_models/conf/callbacks/rollout/tasks/new_playtable_tasks.yaml'))
policy=FrozenHulc(P/'models/HULC_ABC_D',P/'models/HULC_ABC_D/saved_models/HULC_ABC_D.ckpt',P/'data/raw/calvin_debug_dataset/validation/lang_paraphrase-MiniLM-L3-v2/embeddings.npy')
mf=P/'data/data_manifest.json';manifest=json.loads(mf.read_text());records=manifest['records'];start=time.monotonic()
def serial(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,bytes):return x.decode(errors='replace')
 raise TypeError(type(x).__name__)
def digest(obs):return hashlib.sha256(b''.join(obs['rgb_obs'][k].tobytes() for k in sorted(obs['rgb_obs']))+obs['robot_obs'].tobytes()).hexdigest()
def save():
 manifest.update(real_transitions=len(records),status='collecting',records=records,split_counts={s:sum(r['split']==s for r in records) for s in ['train','dev','calibration','test']})
 temp=mf.with_suffix('.tmp');temp.write_text(json.dumps(manifest,indent=2));temp.replace(mf)
def sequence(index,branch):
 c=['red','blue','pink'][index%3];d=['red','blue','pink'][(index+1)%3]
 chains=[['open_drawer',f'lift_{c}_block_table','place_in_drawer','close_drawer','move_slider_left','turn_on_lightbulb'],
  ['move_slider_left',f'lift_{c}_block_slider','place_in_drawer','move_slider_right','turn_on_led','turn_off_led'],
  [f'push_{c}_block_left',f'push_{d}_block_right','open_drawer',f'lift_{d}_block_table','place_in_drawer','close_drawer'],
  ['open_drawer',f'lift_{c}_block_drawer','place_in_slider','close_drawer','turn_off_lightbulb','move_slider_right'],
  ['turn_on_led',f'lift_{c}_block_table','place_in_slider',f'lift_{d}_block_table','place_in_drawer','turn_off_led'],
  ['move_slider_right',f'lift_{c}_block_table','place_in_slider','move_slider_left','turn_on_lightbulb','turn_off_lightbulb']]
 base=chains[index%len(chains)]
 # Preserve the measured prefix for roots already in storage_probe_v2.
 if a.split=='train' and index<3:base=chains[0];base[1]=f'lift_{c}_block_table'
 if branch==0:return base,'demonstration_guided_proposal'
 if branch==1 and index%2==0:return [base[1],base[2],base[0],base[3],base[5],base[4]],'matched_order'
 if branch==2:return base if index%2 else random.Random(9900+index).sample(ids,6),('demonstration_guided_proposal' if index%2 else 'static_catalogue_exploration')
 return random.Random(7100+index).choices(ids,k=6),'static_catalogue_exploration'
def valid_root(root):
 info=root['info'];objects=info['scene_info']['movable_objects'];movable={v['uid'] for v in objects.values()};errors=[]
 for name,obj in objects.items():
  if not np.isfinite(obj['current_pos']).all() or obj['current_pos'][2]<0.35:errors.append(name+':fallen_or_nonfinite')
  if any(c[2] in movable and c[8]<-0.003 for c in obj['contacts']):errors.append(name+':object_interpenetration')
 if any(c[8]<-0.01 and c[9]>0.1 for c in info['robot_info']['contacts']):errors.append('robot_deep_penetration')
 return errors
environments=['A','B'] if a.split=='train' else ['C' if a.split in ['dev','calibration'] else 'D']
for index in range((a.roots+len(environments)-1)//len(environments)):
 for environment in environments:
  if index*len(environments)+environments.index(environment)>=a.roots:continue
  root_id=f'{environment}_{a.split}_root{index:03d}'
  seed={'train':1000,'dev':2000,'calibration':3000,'test':4000}[a.split]+index
  existing=list((P/'data/roots').glob(f'*/{root_id}/full_state_evaluator_only.json'))
  layout=None
  if existing:
   if any(r['root_id']==root_id for r in records):rd=active_root_directory(P,root_id,records)
   elif len(existing)==1:rd=existing[0].parent
   else:raise ValueError('Multiple uncommitted reset roots: '+root_id)
   root=json.loads((rd/'full_state_evaluator_only.json').read_text());identity=json.loads((rd/'identity.json').read_text());layout=identity.get('layout');seed=identity['seed']
  else:
   rd=P/'data/roots/main_v1'/root_id;rd.mkdir(parents=True,exist_ok=False)
   c=['red','blue','pink'][index%3]
   layout={f'block_{color}':'table' for color in ['red','blue','pink']}
   if index%4 in [1,2]:layout['block_'+c]='slider_left' if index%4==1 else 'slider_right'
   accepted=False
   for attempt in range(10):
    actual_seed=seed+10000*attempt;env=CalvinAdapter(P,environment,actual_seed,tabletop=True,segmentation=True,layout=layout)
    try:
     env.reset()
     for door in env.env.scene.doors:door.reset((0.22 if index%2 else 0.0) if door.name=='base__drawer' else (0.28 if index%3 else 0.0))
     for sw in env.env.scene.switches:sw.reset(0.088 if index%4>=2 else 0.0)
     for light in env.env.scene.lights:light.reset(int(index%4>=2) if light.name=='lightbulb' else int(index%2))
     for _ in range(24):env.step(np.array([0,0,0,0,0,0,1.],np.float32))
     # Holding/container initial states are produced only by actual HULC controls.
     init_actions=[]
     if index%5==4:
      init_actions=[f'lift_{c}_block_table']
      if index%10==9:init_actions=['open_drawer']+init_actions+['place_in_drawer']
     for j,skill in enumerate(init_actions):execute_fixed(env,policy,skill,ann[skill][0],budgets[family[skill]],rd/f'initialization_attempt{attempt}_{j}_{skill}',checker)
     root=env.evaluator_snapshot();errors=valid_root(root)
     if errors:
      (rd/f'invalid_attempt{attempt}.json').write_text(json.dumps({'seed':actual_seed,'reasons':errors,'root':root},default=serial));continue
     seed=actual_seed;accepted=True
     np.savez_compressed(rd/'reset_evaluator_only.npz',robot_obs=root['robot_obs'],scene_obs=root['scene_obs'])
     (rd/'full_state_evaluator_only.json').write_text(json.dumps(root,default=serial));OmegaConf.save(env.config,rd/'native_config.yaml')
     (rd/'identity.json').write_text(json.dumps({'root_id':root_id,'split':a.split,'environment':environment,'seed':seed,'layout':layout,
      'initialization_actions':init_actions,'success_not_used_for_root_selection':True,'validity_checks':'finite, no fallen blocks, no deep object/robot interpenetration',
      'guide_sha256':manifest.get('guide_sha256','f4cb908e26db79176263d5002290ac4c94daf3025ce2523fc546eb896e2ad0d4')},indent=2))
     break
    finally:env.close()
   if not accepted:raise RuntimeError('Ten invalid native roots: '+root_id)
  root['robot_obs']=np.asarray(root['robot_obs']);root['scene_obs']=np.asarray(root['scene_obs'])
  baseline_file=rd/'public_root_hash.json';baseline=json.loads(baseline_file.read_text())['sha256'] if baseline_file.exists() else None
  prior=[r for r in records if r['root_id']==root_id and r['step']==0]
  if baseline is None and prior:
   z=np.load(Path(prior[0]['obs_before']).parent.parent/'root_observation_3.npz')
   baseline=digest({'rgb_obs':{k:z[k] for k in ['rgb_static','rgb_gripper']},'robot_obs':z['robot_obs']})
   baseline_file.write_text(json.dumps({'sha256':baseline,'source':'Already recorded root observation'}))
  # Existing measured-prefix root is independently verified by the fresh-instance check.
  for branch in range(a.branches):
   episode=f'{root_id}_branch{branch}';done=[r for r in records if r['episode_id']==episode]
   if len(done)==6:continue
   done=sorted(done,key=lambda r:r['step'])
   if [r['step'] for r in done]!=list(range(len(done))):raise RuntimeError('Partial episode is not a contiguous prefix: '+episode)
   env=CalvinAdapter(P,environment,seed,tabletop=True,segmentation=True,layout=layout)
   try:
    env.reset(root)
    for _ in range(4):obs=env.step(np.array([0,0,0,0,0,0,root['robot_obs'][-1]],np.float32))
    hashed=digest(obs)
    if baseline is None:baseline=hashed;baseline_file.write_text(json.dumps({'sha256':baseline}))
    elif baseline!=hashed:raise RuntimeError('Matched branch reset differs: '+root_id)
    ed=Path(done[0]['obs_before']).parent.parent if done else P/'data/episodes/main_v1'/episode
    if not done:
     ed.mkdir(parents=True,exist_ok=False)
     for j,frame in enumerate(env.history):save_npz(ed/f'root_observation_{j}.npz',frame)
    replay=ed/f'recovery_replay_{len(list(ed.glob("recovery_replay_*"))):02d}' if done else None
    if replay:replay.mkdir()
    actions,style=sequence(index,branch)
    (ed/'reference_evaluator_only.json').write_text(json.dumps({'actions':actions,'style':style,'planner_access':False},indent=2))
    torch.manual_seed(seed*10+branch);np.random.seed(seed*10+branch);random.seed(seed*10+branch)
    for step,skill in enumerate(actions):
     target=(replay if step<len(done) else ed)/f'{step:03d}_{skill}';result=execute_fixed(env,policy,skill,ann[skill][0],budgets[family[skill]],target,checker)
     if step<len(done):
      old=np.load(done[step]['obs_after']);new=np.load(target/'after.npz')
      equal=all(np.array_equal(old[k],new[k]) for k in ['rgb_static','rgb_gripper','robot_obs'])
      (target/'replay_validation.json').write_text(json.dumps({'same_public_endpoint':equal,'original':done[step]['obs_after'],'not_counted_as_new_training_data':True},indent=2))
      if not equal:raise RuntimeError('Paused-prefix replay differs; retain both and inspect: '+episode)
      if pause.requested:
       save();(P/'runs/PAUSED_BY_USER.json').write_text(json.dumps({'status':'paused','stage':'collection_prefix_replay','episode_id':episode,'saved_transitions':len(records),'resume_requires_user_authorization':True},indent=2));raise SystemExit(0)
      continue
     records.append({'episode_id':episode,'root_id':root_id,'split':a.split,'environment':environment,'step':step,'skill':skill,
      'arguments':{'binding_from_static_catalogue':skill},'executor':{'checkpoint':'hulc_abc','budget_steps':budgets[family[skill]]},
      'obs_before':str(target/'before.npz'),'obs_after':str(target/'after.npz'),'low_level_log':str(target/'controls_and_evaluator_state.npz'),
      'labels_path':str(target/'trajectory_evaluator_only.jsonl'),'feedback_source':'offline_evaluator','collection_style':style,
      'success':result['score_at_fixed_budgets'][str(result['requested_steps'])]})
     save();print('TRANSITION',len(records),episode,step,skill,records[-1]['success'],flush=True)
     if pause.requested:
      (P/'runs/PAUSED_BY_USER.json').write_text(json.dumps({'status':'paused','stage':'collection','episode_id':episode,'completed_step':step,'saved_transitions':len(records),'signal':pause.reason,'resume_requires_user_authorization':True},indent=2));print('COLLECTION_PAUSED_SAFELY',len(records),flush=True);raise SystemExit(0)
    if a.stop_after and sum(r['split']==a.split for r in records)>=a.stop_after:
     print('STOP_AFTER_COMPLETE_EPISODE',len(records),flush=True);raise SystemExit(0)
   finally:env.close()
save();print('COLLECTION_COMPLETE',a.split,manifest['split_counts'],time.monotonic()-start,flush=True)
