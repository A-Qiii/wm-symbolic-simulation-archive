"""Persistent native CALVIN state; public IPC responses never contain evaluator truth."""
from pathlib import Path
import argparse,json,time,traceback,hashlib
import numpy as np,torch,hydra
from omegaconf import OmegaConf
from wm_symbolic.env_adapter import CalvinAdapter,save_npz
from wm_symbolic.skill_runner import FrozenHulc,execute_fixed
from wm_symbolic.gpu_phase import gpu_phase
P=Path('/home/robot/research/projects/wm_symbolic');parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--run',required=True);args=parser.parse_args();rd=Path(args.root);run=Path(args.run)
run.mkdir(parents=True,exist_ok=True);(run/'requests').mkdir(exist_ok=True);(run/'responses').mkdir(exist_ok=True);(run/'public').mkdir(exist_ok=True);(run/'evaluator_only').mkdir(exist_ok=True)
def serial(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,bytes):return x.decode(errors='replace')
 raise TypeError(type(x).__name__)
def write(path,data):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,default=serial,indent=2));tmp.replace(path)
identity=json.loads((rd/'identity.json').read_text());root=json.loads((rd/'full_state_evaluator_only.json').read_text());root['robot_obs']=np.asarray(root['robot_obs']);root['scene_obs']=np.asarray(root['scene_obs'])
with gpu_phase(P,'native_environment_creation'):
 env=CalvinAdapter(P,identity['environment'],identity['seed'],tabletop=True,segmentation=True,layout=identity.get('layout'))
ann=OmegaConf.load(P/'vendor/calvin/calvin_models/conf/annotations/new_playtable_validation.yaml');catalog=json.loads((P/'configs/skills.json').read_text());families={x['skill']:x['family'] for x in catalog};budgets=json.loads((P/'configs/executor_budgets_pilot.json').read_text())['steps']
checker=hydra.utils.instantiate(OmegaConf.load(P/'vendor/calvin/calvin_models/conf/callbacks/rollout/tasks/new_playtable_tasks.yaml'))
policy=FrozenHulc(P/'models/HULC_ABC_D',P/'models/HULC_ABC_D/saved_models/HULC_ABC_D.ckpt',P/'data/raw/calvin_debug_dataset/validation/lang_paraphrase-MiniLM-L3-v2/embeddings.npy',device='cpu',reserve_gpu=False)
def public_boundary(name):
 folder=run/'public'/name;folder.mkdir(exist_ok=False);paths=[]
 for i,frame in enumerate(env.history):
  path=folder/f'frame_{i}.npz';save_npz(path,frame);paths.append(str(path))
 assert len(paths)==4
 return paths
try:
 with gpu_phase(P,'native_root_observation'):
  env.reset(root)
  for _ in range(4):env.step(np.array([0,0,0,0,0,0,root['robot_obs'][-1]],np.float32))
 observed=env.observe();public_hash=hashlib.sha256(b''.join(observed['rgb_obs'][k].tobytes() for k in sorted(observed['rgb_obs']))+observed['robot_obs'].tobytes()).hexdigest()
 expected=json.loads((rd/'public_root_hash.json').read_text())['sha256']
 if public_hash!=expected:raise RuntimeError('Executor reset differs from recorded public root')
 write(run/'evaluator_only/reset_validation.json',{'same_public_root':True,'sha256':public_hash})
 execution_index=0
 write(run/'evaluator_only/root_after_observation.json',env.evaluator_snapshot());write(run/'ready.json',{'public_frames':public_boundary('root'),'root_id':identity['root_id'],'hidden_feedback_returned':False})
 request_id=0;last=time.monotonic()
 while True:
  path=run/'requests'/f'{request_id:03d}.json'
  if not path.exists():
   if time.monotonic()-last>900:raise TimeoutError('No planner command for 900 seconds')
   time.sleep(.02);continue
  request=json.loads(path.read_text());command=request['command'];last=time.monotonic()
  if command=='shutdown':write(run/'responses'/f'{request_id:03d}.json',{'closed':True});break
  if command=='skill':
   skill=request['skill']
   if skill not in families:raise ValueError('Skill not in frozen static catalogue')
   target=run/'evaluator_only'/f'{request_id:03d}_{skill}'
   torch.manual_seed(identity['seed']*1000+execution_index);np.random.seed(identity['seed']*1000+execution_index);execution_index+=1
   with gpu_phase(P,'HULC_actual_execution'):
    policy.model.to('cuda:0');policy.device=torch.device('cuda:0')
    try:execute_fixed(env,policy,skill,ann[skill][0],budgets[families[skill]],target,checker)
    finally:policy.reset();policy.model.to('cpu');policy.device=torch.device('cpu');torch.cuda.empty_cache()
   response={'public_frames':public_boundary(f'{request_id:03d}_after_skill'),'control_steps':budgets[families[skill]],'hidden_feedback_returned':False}
  elif command=='reobserve':
   trace=[]
   with gpu_phase(P,'native_fixed_reobservation'):
    for i in range(4):
     obs=env.observe();env.step(np.array([0,0,0,0,0,0,obs['robot_obs'][-1]],np.float32));trace.append(env.evaluator_snapshot())
   write(run/'evaluator_only'/f'{request_id:03d}_reobserve.json',trace)
   response={'public_frames':public_boundary(f'{request_id:03d}_reobserve'),'control_steps':4,'hidden_feedback_returned':False}
  else:raise ValueError('Unknown command')
  write(run/'evaluator_only'/f'{request_id:03d}_boundary.json',env.evaluator_snapshot());write(run/'responses'/f'{request_id:03d}.json',response);request_id+=1
except Exception:
 write(run/'error.json',{'error':traceback.format_exc()});raise
finally:
 with gpu_phase(P,'native_environment_close'):env.close()
