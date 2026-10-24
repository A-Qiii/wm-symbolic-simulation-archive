"""CPU-only parity with official CALVIN wrapper on unchanged recorded images."""
from pathlib import Path
from types import SimpleNamespace
import json,hashlib,datetime,random
import numpy as np,torch
from omegaconf import OmegaConf
from calvin_agent.wrappers.calvin_env_wrapper import CalvinEnvWrapper
from wm_symbolic.skill_runner import FrozenHulc
P=Path('/home/robot/research/projects/wm_symbolic');torch.set_num_threads(2)
policy=FrozenHulc(P/'models/HULC_ABC_D',P/'models/HULC_ABC_D/saved_models/HULC_ABC_D.ckpt',P/'data/raw/calvin_debug_dataset/validation/lang_paraphrase-MiniLM-L3-v2/embeddings.npy',device='cpu',reserve_gpu=False)
wrapper=SimpleNamespace(observation_space_keys=policy.obs_space,transforms=policy.transforms,proprio_state=policy.proprio,device=torch.device('cpu'))
manifest=json.loads((P/'data/data_manifest.json').read_text());chosen=[]
for name in ['open_drawer','lift_red_block_table','lift_blue_block_table','lift_pink_block_table','place_in_drawer','turn_on_led']:
 chosen.append(next(r for r in manifest['records'] if r['skill']==name))
annotations=OmegaConf.load(P/'vendor/calvin/calvin_models/conf/annotations/new_playtable_validation.yaml');real_model=policy.model;rows=[]
class Capture:
 def step(self,x,command):self.input=x;return torch.zeros(1,7)
def flatten(x,prefix=''):
 result={}
 for k,v in x.items():
  if isinstance(v,dict):result.update(flatten(v,prefix+k+'.'))
  elif torch.is_tensor(v):result[prefix+k]=v
 return result
for row in chosen:
 with np.load(row['obs_before']) as z:obs={'rgb_obs':{k:z[k] for k in ['rgb_static','rgb_gripper']},'depth_obs':{k:z[k] for k in ['depth_static','depth_gripper']},'robot_obs':z['robot_obs']}
 command=annotations[row['skill']][0];official=CalvinEnvWrapper.transform_observation(wrapper,obs);capture=Capture();policy.model=capture;policy.step(obs,command);adapted=capture.input;policy.model=real_model
 left=flatten(official);right=flatten(adapted);assert left.keys()==right.keys();deltas={k:float((left[k]-right[k]).abs().max()) for k in left};assert all(v==0 for v in deltas.values())
 def seed():torch.manual_seed(1783);np.random.seed(1783);random.seed(1783);real_model.reset()
 seed()
 with torch.inference_mode():native_action=real_model.step(official,command).squeeze().numpy().copy()
 native_action[-1]=1 if native_action[-1]>0 else -1
 seed();adapter_action=policy.step(obs,command);difference=float(np.abs(native_action-adapter_action).max());assert difference==0,difference
 rows.append({'skill':row['skill'],'input':row['obs_before'],'source_root':row['root_id'],'input_tensor_shapes':{k:list(v.shape) for k,v in left.items()},'preprocessing_max_abs_difference':deltas,'seeded_first_action_max_abs_difference':difference,'command':command})
 print('OFFICIAL_PARITY',row['skill'],difference,flush=True)
native=OmegaConf.to_container(OmegaConf.load(P/'data/roots/main_v1/C_dev_root000/native_config.yaml'),resolve=True);official=OmegaConf.to_container(OmegaConf.load(P/'data/raw/calvin_debug_dataset/validation/.hydra/merged_config.yaml'),resolve=True)
camera_equal={k:native['cameras'][k]==official['cameras'][k] for k in ['static','gripper']};assert all(camera_equal.values())
result={'passed':True,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'device':'cpu','simulation_launched':False,'policy_updated':False,'source_wrapper':str(P/'vendor/calvin/calvin_models/calvin_agent/wrappers/calvin_env_wrapper.py'),'wrapper_sha256':hashlib.sha256((P/'vendor/calvin/calvin_models/calvin_agent/wrappers/calvin_env_wrapper.py').read_bytes()).hexdigest(),'rows':rows,'native_vs_official_camera_configs_equal':camera_equal,'official_saved_scene':official['scene']['name'],'limitation':'Identical preprocessing and seeded first actions on the same recorded observations do not establish successful rollout. Standard official initial-state physical reproduction remains necessary.','compatibility_adaptations':policy.identity}
(P/'runs/executor_preprocessing_parity.json').write_text(json.dumps(result,indent=2));print('PREPROCESSING_PARITY_COMPLETE',len(rows))
