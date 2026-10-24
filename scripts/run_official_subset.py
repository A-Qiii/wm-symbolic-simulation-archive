"""Isolated official-standard 20-sequence health check; not main D evaluation.

Uses CALVIN's unmodified rollout() also imported by HULC's evaluation entry.
Strict released-checkpoint compatibility loader is FrozenHulc; no WM/stop layer.
Official success feedback terminates ONLY this diagnostic subtask.
"""
from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict
import sys,json,random,hashlib,datetime,time,os
P=Path('/home/robot/research/projects/wm_symbolic')
sys.path.insert(0,str(P/'scripts/official_compat'))
import numpy as np,torch,hydra,imageio.v2 as imageio
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from calvin_agent.evaluation.evaluate_policy import rollout,EP_LEN
from calvin_agent.evaluation.utils import get_env_state_for_initial_condition
from calvin_agent.wrappers.calvin_env_wrapper import CalvinEnvWrapper
from wm_symbolic.skill_runner import FrozenHulc
from wm_symbolic.env_adapter import save_npz,CalvinAdapter
from wm_symbolic.checkpointing import PauseRequest
out=P/'runs/official_standard_subset_20261014';protocol=json.loads((out/'protocol.json').read_text());assert protocol['status']=='prepared_not_run'
assert not (out/'results.json').exists(),'Never overwrite/reselect a completed or partial diagnostic'
assert EP_LEN==360
seed_everything(protocol['policy_random_seed'],workers=True);torch.set_num_threads(4);pause=PauseRequest()
policy=FrozenHulc(P/'models/HULC_ABC_D',P/'models/HULC_ABC_D/saved_models/HULC_ABC_D.ckpt',P/'data/raw/calvin_debug_dataset/validation/lang_paraphrase-MiniLM-L3-v2/embeddings.npy')
assert not any(p.requires_grad for p in policy.model.parameters())
assert not policy.model.training and policy.model.replan_freq==30
assert not policy.model.perceptual_encoder.proprio_encoder
assert policy.model.action_decoder.gripper_control
class PauseDiagnostic(Exception):pass
def serial(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,bytes):return x.decode(errors='replace')
 raise TypeError(type(x).__name__)
class RecordedOfficialWrapper(CalvinEnvWrapper):
 @staticmethod
 def set_egl_device(device):
  # The official auto-probe requires an absent C compiler. Reuse the explicit
  # EGL device already validated by all prior local RTX5080 CALVIN rollouts.
  if device.type!='cuda' or device.index!=0 or os.environ.get('EGL_VISIBLE_DEVICES')!='0':
   raise RuntimeError('Only the already verified local cuda0/EGL0 mapping is permitted')
 def transform_observation(self,raw):
  self.raw=raw
  transformed=super().transform_observation(raw)
  assert np.array_equal(transformed['robot_obs_raw'].cpu().numpy(),raw['robot_obs'])
  return transformed
 def begin(self,folder,skill):
  self.folder=folder;folder.mkdir();self.active=True;self.skill=skill
  raw=self.env.get_obs();self.raw=raw;save_npz(folder/'before.npz',CalvinAdapter.model_view(raw))
  self.initial=self.env.get_info();self.actions=[];self.robot=[];self.scene=[];self.infos=[self.initial]
  self.writers={k:imageio.get_writer(str(folder/f'{k}.mp4'),fps=self.env.control_freq,codec='libx264',macro_block_size=1,quality=7) for k in raw['rgb_obs']}
  for k,w in self.writers.items():w.append_data(raw['rgb_obs'][k])
  self.rows=(folder/'trajectory_evaluator_only.jsonl').open('w');self.rows.write(json.dumps({'control_step':0,'info':self.initial},default=serial)+'\n')
 def step(self,action):
  if pause.requested:raise PauseDiagnostic()
  result=super().step(action)
  control=action.detach().squeeze().cpu().numpy().copy();control[-1]=1 if control[-1]>0 else -1
  self.actions.append(control);self.robot.append(self.raw['robot_obs'].copy());self.scene.append(self.raw['scene_obs'].copy());self.infos.append(result[3])
  for k,w in self.writers.items():w.append_data(self.raw['rgb_obs'][k])
  self.rows.write(json.dumps({'control_step':len(self.actions),'info':result[3]},default=serial)+'\n')
  return result
 def finish(self,success,exception=None):
  if not getattr(self,'active',False):return None
  self.active=False
  for w in self.writers.values():w.close()
  self.rows.close();save_npz(self.folder/'after.npz',CalvinAdapter.model_view(self.raw))
  np.savez_compressed(self.folder/'controls_and_evaluator_state.npz',actions=self.actions,robot_obs=self.robot,scene_obs=self.scene)
  summary={'skill':self.skill,'controls':len(self.actions),'success':success,'terminal_reason':'interrupted' if exception else 'official_success' if success else 'official_360_timeout','exception':exception,'simulated_seconds':len(self.actions)/self.env.control_freq,'source':'unmodified calvin_agent.evaluation.evaluate_policy.rollout','official_success_stopping_used':True,'world_model_used':False,'visual_stopper_used':False,'video_frames':1+len(self.actions),'video_contains_initial_frame':True,'run':str(self.folder)}
  (self.folder/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));return summary
dataset=SimpleNamespace(abs_datasets_dir=P/'data/raw/calvin_debug_dataset/validation',observation_space=policy.obs_space,transforms=policy.transforms,proprio_state=policy.proprio)
env=RecordedOfficialWrapper(dataset,policy.device,show_gui=False);env.active=False
checker=hydra.utils.instantiate(OmegaConf.load(P/'vendor/calvin/calvin_models/conf/callbacks/rollout/tasks/new_playtable_tasks.yaml'))
commands=OmegaConf.load(P/'vendor/calvin/calvin_models/conf/annotations/new_playtable_validation.yaml')
manifest=P/'data/data_manifest.json';manifest_sha=hashlib.sha256(manifest.read_bytes()).hexdigest()
source_paths=['vendor/hulc/hulc/evaluation/evaluate_policy.py','vendor/calvin/calvin_models/calvin_agent/evaluation/evaluate_policy.py','vendor/calvin/calvin_models/calvin_agent/evaluation/utils.py','vendor/calvin/calvin_models/calvin_agent/evaluation/multistep_sequences.py','vendor/calvin/calvin_models/calvin_agent/wrappers/calvin_env_wrapper.py','src/wm_symbolic/skill_runner.py']
record={'status':'running','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Official-standard first20 subset health check, not full1000 benchmark and not mainD test','protocol':str(out/'protocol.json'),'checkpoint':policy.identity,'model_trainable_parameter_count':sum(p.numel() for p in policy.model.parameters() if p.requires_grad),'source_sha256':{f:hashlib.sha256((P/f).read_bytes()).hexdigest() for f in source_paths},'official_rollout_reused':True,'official_launcher_used':False,'loader_compatibility':'Strict frozen loader with recorded historical transform aliases, antialiasFalse, dormant LayerNorm restoration','EGL_compatibility':'Explicit previously verified CUDA0/EGL0 mapping; official compiled auto-probe unavailable, no camera or action transformation change','raw_robot_pose_passthrough_asserted_each_frame':True,'main_manifest_sha256_before':manifest_sha,'sequences':[]}
def save():
 tmp=out/'results.tmp';tmp.write_text(json.dumps(record,ensure_ascii=False,indent=2,default=serial));tmp.replace(out/'results.json')
save();plans=defaultdict(list)
try:
 for spec in protocol['selected']:
  index=spec['index'];folder=out/f'sequence_{index:04d}';robot,scene=get_env_state_for_initial_condition(spec['initial_condition'])
  with np.load(spec['reset_npz']) as z:assert np.array_equal(robot,z['robot_obs']) and np.array_equal(scene,z['scene_obs'])
  torch.save({'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all(),'numpy':np.random.get_state(),'python':random.getstate()},folder/'rng_before_reset.pt')
  env.reset(robot_obs=robot,scene_obs=scene)
  raw=env.raw;save_npz(folder/'actual_reset_public.npz',CalvinAdapter.model_view(raw));np.savez_compressed(folder/'actual_reset_evaluator_only.npz',robot_obs=raw['robot_obs'],scene_obs=raw['scene_obs'])
  (folder/'full_reset_serialization_evaluator_only.json').write_text(json.dumps({'serialization':env.env.serialize(),'info':env.get_info()},default=serial))
  differences={'robot_max_abs':float(np.max(np.abs(raw['robot_obs']-robot))),'robot_last_gripper_action_expected_actual':[float(robot[-1]),float(raw['robot_obs'][-1])],'scene_max_abs':float(np.max(np.abs(raw['scene_obs']-scene)))}
  seq={'index':index,'skills':spec['skills'],'initial_condition':spec['initial_condition'],'reset_differences':differences,'subtasks':[],'successful_prefix':0}
  record['sequences'].append(seq);save()
  for j,skill in enumerate(spec['skills']):
   env.begin(folder/f'{j:02d}_{skill}',skill);success=None
   try:
    with torch.inference_mode():success=bool(rollout(env,policy.model,checker,skill,commands,plans,False))
   except BaseException as e:
    item=env.finish(False,repr(e));seq['subtasks'].append(item);save();raise
   item=env.finish(success);seq['subtasks'].append(item);seq['successful_prefix']+=int(success);save()
   print('OFFICIAL_SUBSET',index,j,skill,item['controls'],success,flush=True)
   if not success:break
  seq['complete']=True;save()
except PauseDiagnostic:
 record['status']='paused';record['pause_reason']=pause.reason;save()
except BaseException as e:
 record['status']='error';record['error']=repr(e);save();raise
else:
 record.update(status='complete',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),completed_sequences=len(record['sequences']),total_subtasks=sum(len(s['subtasks']) for s in record['sequences']),successful_subtasks=sum(s['successful_prefix'] for s in record['sequences']))
finally:
 if env.env.ownsPhysicsClient:env.env.close();env.env.ownsPhysicsClient=False
 record['main_manifest_sha256_after']=hashlib.sha256(manifest.read_bytes()).hexdigest()
 record['checkpoint_sha256_after']=hashlib.sha256((P/'models/HULC_ABC_D/saved_models/HULC_ABC_D.ckpt').read_bytes()).hexdigest()
 record['main_pause_preserved']=(P/'runs/PAUSED_BY_USER.json').exists() and (P/'runs/FORMAL_TRAINING_HOLD.json').exists()
 assert record['main_manifest_sha256_after']==manifest_sha
 assert record['checkpoint_sha256_after']==policy.identity['checkpoint_sha256']
 save()
print('OFFICIAL_SUBSET_FINAL',record['status'],record.get('completed_sequences'),record.get('successful_subtasks'),flush=True)
