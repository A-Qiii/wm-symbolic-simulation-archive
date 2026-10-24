"""First 50 real training transitions; finish the current six-step episode at 54.

This is data collection only: no world-model training, tuning on D, or invented
outcomes. Roots/branches remain grouped. RGB, depth, visible segmentation and
calibration are public; scene/contact trajectories remain evaluator-only.
"""
from pathlib import Path
import json,time,random,hashlib,sys,os
import numpy as np,torch,hydra
from omegaconf import OmegaConf
from wm_symbolic.env_adapter import CalvinAdapter,save_npz
from wm_symbolic.skill_runner import FrozenHulc,execute_fixed
P=Path('/home/robot/research/projects/wm_symbolic')
OUT=P/'data/episodes/storage_probe_v2';OUT.mkdir(exist_ok=False)
ROOTS=P/'data/roots/storage_probe_v2';ROOTS.mkdir(exist_ok=False)
old=P/'data/data_manifest.json'
if old.exists():
 archived=json.loads(old.read_text());archived['status']='retained_unpaired_reset_diagnostic';archived['exclusion_reason']='First engine reused after physical actions; paired reset equality failed. Six valid observations retained separately from the controlled pilot.'
 (P/'data/storage_probe_v1_manifest.json').write_text(json.dumps(archived,indent=2))
ann=OmegaConf.load(P/'vendor/calvin/calvin_models/conf/annotations/new_playtable_validation.yaml')
catalog=json.loads((P/'configs/skills.json').read_text());ids=[s['skill'] for s in catalog]
checker=hydra.utils.instantiate(OmegaConf.load(P/'vendor/calvin/calvin_models/conf/callbacks/rollout/tasks/new_playtable_tasks.yaml'))
policy=FrozenHulc(P/'models/HULC_ABC_D',P/'models/HULC_ABC_D/saved_models/HULC_ABC_D.ckpt',P/'data/raw/calvin_debug_dataset/validation/lang_paraphrase-MiniLM-L3-v2/embeddings.npy')
# Evaluated budget choices; unsupported bindings stay in the catalogue.
budgets={'drawer':240,'slider':120,'device':120,'push':120,'lift':360,'place':360}
family={s['skill']:s['family'] for s in catalog}
(P/'configs/executor_budgets_pilot.json').write_text(json.dumps({'steps':budgets,'frozen_for':'storage_probe_v2',
 'source':'C baseline + colour contrast + container reference; no D roots',
 'limitations':'Red/blue table lift not validated in C; failures retained. Place/lift use 360 tested steps.'},indent=2))
records=[];start_time=time.monotonic()
def serial(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,bytes):return x.decode(errors='replace')
 raise TypeError(type(x).__name__)
def digest(obs):return hashlib.sha256(b''.join(obs['rgb_obs'][k].tobytes() for k in sorted(obs['rgb_obs']))+obs['robot_obs'].tobytes()).hexdigest()
def sequence(index,branch):
 color=['red','blue','pink'][index%3]
 base=['open_drawer',f'lift_{color}_block_table','place_in_drawer','close_drawer','move_slider_left','turn_on_lightbulb']
 if branch==0:return base,'reference_proposal'
 if index%2==0:return [base[1],base[2],base[0],base[3],base[5],base[4]],'matched_order'
 return random.Random(7100+index).choices(ids,k=6),'static_catalogue_exploration'
def save_manifest():
 (P/'data/data_manifest.json').write_text(json.dumps({'schema':'macro_v1','status':'storage_probe_only',
  'real_transitions':len(records),'records':records,'root_split':'A/B train only',
  'counts_not_synthetic':True,'training_run':False,'masks':'Visible PyBullet object/link pixels; diagnostic condition, not open detection.',
  'protocol':str(P/'configs/protocol.json'),'retained_diagnostic_transitions':'data/storage_probe_v1_manifest.json',
  'reset_protocol':'fresh native physics instance for every branch; saved official root; four public observation steps'},indent=2))
for index in range(3):
 for environment in ['A','B']:
  env=CalvinAdapter(P,environment,1000+index,tabletop=True,segmentation=True)
  root_id=f'{environment}_train_root{index:03d}';rd=ROOTS/root_id;rd.mkdir()
  try:
   env.reset()
   for _ in range(12):env.step(np.array([0,0,0,0,0,0,1.],np.float32))
   root=env.evaluator_snapshot()
   # Reject visible physical breakdown instead of training on invalid roots.
   objects=root['info']['scene_info']['movable_objects']
   if any(not np.isfinite(v['current_pos']).all() or v['current_pos'][2]<0.35 for v in objects.values()):
    (rd/'invalid.json').write_text(json.dumps({'reason':'nonfinite_or_fallen_object'},indent=2));raise RuntimeError(root_id+' invalid root')
   np.savez_compressed(rd/'reset_evaluator_only.npz',robot_obs=root['robot_obs'],scene_obs=root['scene_obs'])
   (rd/'full_state_evaluator_only.json').write_text(json.dumps(root,default=serial))
   OmegaConf.save(env.config,rd/'native_config.yaml')
   (rd/'identity.json').write_text(json.dumps({'root_id':root_id,'split':'train','environment':environment,'seed':1000+index,
    'initialization':'native tabletop sampling followed by 12 zero-relative control steps; gripper open',
    'guide_sha256':'f4cb908e26db79176263d5002290ac4c94daf3025ce2523fc546eb896e2ad0d4'},indent=2))
   baseline=None
   for branch in range(2):
    env.close()
    env=CalvinAdapter(P,environment,1000+index,tabletop=True,segmentation=True)
    obs=env.reset(root)
    # The public four-frame pre-observation protocol is the same for branches.
    for _ in range(4):obs=env.step(np.array([0,0,0,0,0,0,1.],np.float32))
    current_digest=digest(obs)
    if baseline is None:baseline=current_digest
    elif current_digest!=baseline:
     (rd/'branch_reset_mismatch.json').write_text(json.dumps({'first':baseline,'second':current_digest},indent=2))
     raise RuntimeError('Matched branch root differs after observation protocol: '+root_id)
    episode=f'{root_id}_branch{branch}';ed=OUT/episode;ed.mkdir()
    for j,frame in enumerate(env.history):save_npz(ed/f'root_observation_{j}.npz',frame)
    actions,style=sequence(index,branch)
    (ed/'reference_evaluator_only.json').write_text(json.dumps({'actions':actions,'style':style,'planner_access':False},indent=2))
    torch.manual_seed(5000+index*2+branch);random.seed(5000+index*2+branch);np.random.seed(5000+index*2+branch)
    for step,skill in enumerate(actions):
     target=ed/f'{step:03d}_{skill}'
     result=execute_fixed(env,policy,skill,ann[skill][0],budgets[family[skill]],target,checker)
     records.append({'episode_id':episode,'root_id':root_id,'split':'train','environment':environment,'step':step,
      'skill':skill,'arguments':{'binding_from_static_catalogue':skill},'executor':{'checkpoint':'hulc_abc','budget_steps':budgets[family[skill]]},
      'obs_before':str(target/'before.npz'),'obs_after':str(target/'after.npz'),
      'low_level_log':str(target/'controls_and_evaluator_state.npz'),'labels_path':str(target/'trajectory_evaluator_only.jsonl'),
      'feedback_source':'offline_evaluator','collection_style':style,'success':result['score_at_fixed_budgets'][str(result['requested_steps'])]})
     save_manifest();print('TRANSITION',len(records),episode,step,skill,records[-1]['success'],flush=True)
     if len(records)==50:
      files=[f for f in OUT.rglob('*') if f.is_file()]
      groups={'video_bytes':sum(f.stat().st_size for f in files if f.suffix=='.mp4'),
              'npz_bytes':sum(f.stat().st_size for f in files if f.suffix=='.npz'),
              'evaluator_trace_bytes':sum(f.stat().st_size for f in files if f.suffix=='.jsonl'),
              'other_bytes':sum(f.stat().st_size for f in files if f.suffix not in {'.mp4','.npz','.jsonl'})}
      total=sum(groups.values())
      (P/'runs/storage_50.json').write_text(json.dumps({'count':50,'actual_bytes':total,'groups':groups,
       'estimated_6000_trajectory_bytes':total/50*6000,'estimate_excludes':'Frozen DINO feature cache and model checkpoints',
       'elapsed_seconds':time.monotonic()-start_time,'root_leakage':'A/B train only; paired branches same split'},indent=2))
    if len(records)>=50:
     print('Finished current six-step sequence; collected',len(records),flush=True);sys.exit(0)
  finally:env.close()
