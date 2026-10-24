"""Frozen HULC on unchanged official debug initial states, two inference paths."""
from pathlib import Path
from collections import deque
from copy import deepcopy
from types import SimpleNamespace
import json,time,hashlib,random,datetime
import numpy as np,torch,hydra
from omegaconf import OmegaConf
from calvin_env.envs.play_table_env import get_env
from calvin_agent.wrappers.calvin_env_wrapper import CalvinEnvWrapper
from wm_symbolic.skill_runner import FrozenHulc,execute_fixed
from wm_symbolic.env_adapter import CalvinAdapter
from wm_symbolic.checkpointing import PauseRequest
P=Path('/home/robot/research/projects/wm_symbolic');out=P/'runs/executor_official_reproduction_v1';out.mkdir(exist_ok=False);pause=PauseRequest();torch.set_num_threads(4)
dataset=P/'data/raw/calvin_debug_dataset/validation';ann=np.load(dataset/'lang_paraphrase-MiniLM-L3-v2/auto_lang_ann.npy',allow_pickle=True).item();commands=OmegaConf.load(P/'vendor/calvin/calvin_models/conf/annotations/new_playtable_validation.yaml');checker=hydra.utils.instantiate(OmegaConf.load(P/'vendor/calvin/calvin_models/conf/callbacks/rollout/tasks/new_playtable_tasks.yaml'))
policy=FrozenHulc(P/'models/HULC_ABC_D',P/'models/HULC_ABC_D/saved_models/HULC_ABC_D.ckpt',dataset/'lang_paraphrase-MiniLM-L3-v2/embeddings.npy')
wrapper=SimpleNamespace(observation_space_keys=policy.obs_space,transforms=policy.transforms,proprio_state=policy.proprio,device=policy.device)
class OfficialPolicy:
 def __init__(self,base):self.base=base;self.identity=dict(base.identity,inference_path='Official CalvinEnvWrapper.transform_observation followed by unchanged Hulc.step')
 def reset(self):self.base.reset()
 def step(self,obs,command):
  x=CalvinEnvWrapper.transform_observation(wrapper,obs)
  with torch.inference_mode():action=self.base.model.step(x,command).squeeze().cpu().numpy()
  action[-1]=1 if action[-1]>0 else -1
  return action
class OfficialAdapter:
 def __init__(self,robot,scene):
  self.env=get_env(dataset,obs_space=policy.obs_space,show_gui=False);self.history=deque(maxlen=4);raw=self.env.reset(robot_obs=robot,scene_obs=scene);self.history.append(CalvinAdapter.model_view(raw))
 def observe(self):return CalvinAdapter.model_view(self.env.get_obs())
 def step(self,action):
  raw,_,_,_=self.env.step(action);public=CalvinAdapter.model_view(raw);self.history.append(public);return public
 def evaluator_snapshot(self):return {'environment':'D_official_debug_diagnostic',**deepcopy(self.env.get_state_obs()),'info':deepcopy(self.env.get_info())}
 def close(self):
  if self.env.ownsPhysicsClient:self.env.close();self.env.ownsPhysicsClient=False
record={'status':'running','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'diagnostic_only':True,'world_model_training_or_selection':False,'official_annotations':len(ann['language']['task']),'seeds':[0,1,2],'steps':360,'GT_stopping':False,'paths':['official_wrapper','project_adapter'],'policy_changed':False,'trials':[],'paired_results':[]}
def save():
 tmp=out/'results.tmp';tmp.write_text(json.dumps(record,ensure_ascii=False,indent=2));tmp.replace(out/'results.json')
save()
for index,skill in enumerate(ann['language']['task']):
 start,end=map(int,ann['info']['indx'][index]);source=dataset/f'episode_{start:07d}.npz'
 with np.load(source) as z:robot=z['robot_obs'].copy();scene=z['scene_obs'].copy()
 for seed in [0,1,2]:
  pair=[]
  for mode in ['official_wrapper','project_adapter']:
   env=OfficialAdapter(robot,scene)
   try:
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);folder=out/f'{index:02d}_{skill}_seed{seed}_{mode}';r=execute_fixed(env,OfficialPolicy(policy) if mode=='official_wrapper' else policy,skill,commands[skill][0],360,folder,checker)
    infos=[json.loads(line)['info'] for line in (folder/'trajectory_evaluator_only.jsonl').read_text().splitlines()];hit=[i for i in range(1,len(infos)) if skill in checker.get_task_info_for_set(infos[0],infos[i],{skill})]
    before=np.load(folder/'before.npz');after=np.load(folder/'after.npz');controls=np.load(folder/'controls_and_evaluator_state.npz');public_hash=lambda z:hashlib.sha256(b''.join(z[k].tobytes() for k in ['rgb_static','rgb_gripper','robot_obs'])).hexdigest()
    summary={k:v for k,v in r.items() if k!='executor'};summary.update(annotation_index=index,official_source=str(source),annotated_end=end,seed=seed,inference_path=mode,first_success_control_step=hit[0] if hit else None,success_at_any_step=bool(hit),success_at_budget_end=bool(r['score_at_fixed_budgets']['360']),public_root_sha256=public_hash(before),public_end_sha256=public_hash(after),actions_sha256=hashlib.sha256(controls['actions'].tobytes()).hexdigest(),run=str(folder));record['trials'].append(summary);pair.append(summary);save();print('OFFICIAL_REPRODUCTION',index,skill,seed,mode,summary['first_success_control_step'],summary['success_at_budget_end'],flush=True)
   finally:env.close()
   if pause.requested:record['status']='paused';save();raise SystemExit(0)
  same={key:pair[0][key]==pair[1][key] for key in ['public_root_sha256','public_end_sha256','actions_sha256']};record['paired_results'].append({'annotation_index':index,'skill':skill,'seed':seed,**same});save()
record.update(status='complete',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),all_paired_trajectories_identical=all(all(x[k] for k in ['public_root_sha256','public_end_sha256','actions_sha256']) for x in record['paired_results']));save();print('OFFICIAL_REPRODUCTION_COMPLETE',len(record['trials']),record['all_paired_trajectories_identical'],flush=True)
