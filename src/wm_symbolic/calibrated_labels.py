"""Correct table regions in the native table's physical coordinate system."""
from pathlib import Path
from functools import lru_cache
import hashlib
import numpy as np
from wm_symbolic.state_labels import labels_at,SPEC,COLORS
DATA=Path('/home/robot/research/projects/wm_symbolic/vendor/calvin/calvin_env/data')
@lru_cache(maxsize=4)
def mesh_x_bounds(urdf,scale):
 mesh=(DATA/urdf).parent.parent/'meshes/base_link.obj';vertices=[]
 for line in mesh.open():
  if line.startswith('v '):vertices.append([float(x)*scale for x in line.split()[1:4]])
 v=np.asarray(vertices);return float(v[:,0].min()),float(v[:,0].max()),str(mesh),hashlib.sha256(mesh.read_bytes()).hexdigest()
def calibrated_labels_at(info,scene_cfg):
 label=labels_at(info,scene_cfg);table=scene_cfg['objects']['fixed_objects']['table'];scale=float(scene_cfg['global_scaling']);lo,hi,mesh,sha=mesh_x_bounds(table['file'],scale)
 # Native A/B/C/D table frames have zero orientation. Reject an unsupported pose.
 if not np.allclose(table['initial_orn'],[0,0,0]):raise ValueError('Native table frame rotation needs explicit calibrated transform')
 origin=np.asarray(table['initial_pos']);cuts=np.linspace(lo,hi,4)[1:3]
 for j,color in enumerate(COLORS):
  if label['categorical'][j]<3:
   local=np.asarray(info['scene_info']['movable_objects']['block_'+color]['current_pos'])-origin;x=local[0]
   label['categorical'][j]=int(x>=cuts[0])+int(x>=cuts[1]);label['categorical_valid'][j]=bool(min(abs(x-cuts[0]),abs(x-cuts[1]))>SPEC['boundary_margin_m'])
   label['continuous']['objects'][color]['table_local_position_m']=local.tolist()
 label['table_calibration']={'mesh':mesh,'mesh_sha256':sha,'local_x_bounds_m':[lo,hi],'local_x_thirds_m':cuts.tolist(),'source':'Physical native table mesh, not object-placement sampling bounds'}
 return label
