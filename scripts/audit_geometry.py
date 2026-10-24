from pathlib import Path
import json,yaml
import numpy as np
from wm_symbolic.calibrated_labels import mesh_x_bounds
P=Path('/home/robot/research/projects/wm_symbolic');rows=[]
for env in ['A','B','C']:
 cfg=yaml.safe_load((P/'vendor/calvin/calvin_env/conf/scene'/f'calvin_scene_{env}.yaml').read_text());lo,hi,path,sha=mesh_x_bounds(cfg['objects']['fixed_objects']['table']['file'],cfg['global_scaling'])
 old=np.linspace(cfg['surfaces']['table'][0][0],cfg['surfaces']['table'][1][0],4)[1:3];new=np.linspace(lo,hi,4)[1:3]
 rows.append({'environment':env,'old_sampling_region_cuts_m':old.tolist(),'physical_mesh_cuts_m':new.tolist(),'mesh':path,'sha256':sha,'same_local_x_0.05_old_class':int(.05>=old[0])+int(.05>=old[1]),'same_local_x_0.05_corrected_class':int(.05>=new[0])+int(.05>=new[1])})
assert len({tuple(r['physical_mesh_cuts_m']) for r in rows})==1
(P/'runs/label_geometry_audit.json').write_text(json.dumps({'evidence':rows,'source_of_error':'Placement sampling rectangles were mistakenly used as physical tabletop calibration','repair':'Use actual native mesh/table-local coordinates consistently; no threshold optimized on model/test outcomes','test_trajectories_used':False},indent=2));print(json.dumps(rows))
