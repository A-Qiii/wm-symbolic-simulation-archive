from pathlib import Path
import json
import numpy as np
from wm_symbolic.env_adapter import CalvinAdapter,save_npz
P=Path('/home/robot/research/projects/wm_symbolic')
rows=[]
for letter in ['A','B']:
 env=CalvinAdapter(P,letter,1000,tabletop=True,segmentation=True)
 try:
  obs=env.reset()
  for _ in range(12):obs=env.step(np.array([0,0,0,0,0,0,1.],np.float32))
  root=env.evaluator_snapshot()
  for camera,mask in obs['segmentation'].items():
   assert mask.shape==obs['rgb_obs']['rgb_'+camera].shape[:2]
   assert mask.dtype==np.int32 and len(np.unique(mask))>1
   assert len(obs['calibration'][camera]['viewMatrix'])==16
   assert len(obs['calibration'][camera]['projectionMatrix'])==16
  a=env.reset(root)
  for _ in range(4):a=env.step(np.array([0,0,0,0,0,0,1.],np.float32))
  for _ in range(12):env.step(np.array([0,0,0,0,0,0,1.],np.float32))
  b=env.reset(root)
  for _ in range(4):b=env.step(np.array([0,0,0,0,0,0,1.],np.float32))
  errors={k:int(np.abs(a['rgb_obs'][k].astype(int)-b['rgb_obs'][k].astype(int)).max()) for k in a['rgb_obs']}
  errors['robot']=float(np.abs(a['robot_obs']-b['robot_obs']).max())
  assert all(v==0 for v in errors.values()),errors
  save_npz(P/'runs'/f'masks_{letter}.npz',a)
  rows.append({'environment':letter,'mask_counts':{k:len(np.unique(v)) for k,v in a['segmentation'].items()},'reset_max_abs_errors':errors,
   'public_fields':list(a),'hidden_fields_absent':not {'scene_obs','info','success','contacts'}&set(a),
   'mask_semantics':'Only visible renderer pixels, anonymous body/link IDs; diagnostic condition','passed':True})
 finally:env.close()
(P/'runs/mask_probe.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows),flush=True)
