from pathlib import Path
import json
import numpy as np
from wm_symbolic.env_adapter import CalvinAdapter,save_npz
P=Path('/home/robot/research/projects/wm_symbolic');R=P/'data/roots/storage_probe_v1/A_train_root000'
root=json.loads((R/'full_state_evaluator_only.json').read_text())
root['robot_obs']=np.asarray(root['robot_obs']);root['scene_obs']=np.asarray(root['scene_obs'])
def start(env):
 x=env.reset(root)
 for _ in range(4):x=env.step(np.array([0,0,0,0,0,0,1.],np.float32))
 return x
def diff(a,b):
 return {'rgb_max':{k:int(np.abs(a['rgb_obs'][k].astype(int)-b['rgb_obs'][k].astype(int)).max()) for k in a['rgb_obs']},
 'rgb_changed_pixels':{k:int((a['rgb_obs'][k]!=b['rgb_obs'][k]).any(-1).sum()) for k in a['rgb_obs']},
 'robot_abs':np.abs(a['robot_obs']-b['robot_obs']).tolist()}
env=CalvinAdapter(P,'A',1000,tabletop=True,segmentation=True)
try:
 a=start(env);sa=env.evaluator_snapshot()
 for f in sorted((P/'data/episodes/storage_probe_v1/A_train_root000_branch0').glob('*/controls_and_evaluator_state.npz')):
  for action in np.load(f)['actions']:env.step(action)
 b=start(env);sb=env.evaluator_snapshot()
finally:env.close()
env=CalvinAdapter(P,'A',1000,tabletop=True,segmentation=True)
try:c=start(env)
finally:env.close()
result={'same_engine_after_actual_controls':diff(a,b),'fresh_engine_same_saved_root':diff(a,c),
 'objects_after':{n:{k:v for k,v in o.items() if k in ['current_pos','current_lin_vel','current_ang_vel']} for n,o in sb['info']['scene_info']['movable_objects'].items()},
 'objects_before':{n:{k:v for k,v in o.items() if k in ['current_pos','current_lin_vel','current_ang_vel']} for n,o in sa['info']['scene_info']['movable_objects'].items()},
 'replay':'Actual recorded low-level controls, unchanged root and public observation protocol'}
(P/'runs/reset_after_actions.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
