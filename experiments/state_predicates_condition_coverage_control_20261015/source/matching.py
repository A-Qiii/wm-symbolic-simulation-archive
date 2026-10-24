"""Predeclared pose matching and scoring; reads saved outputs, no model inference."""
from experiment import *
import numpy as np,csv
from scipy.spatial.transform import Rotation
def known(p):
 k=np.isfinite(p);positive=p[:3]>=.5
 if positive.sum()>1:k[:3]&=~positive
 return k
def ternary(p):
 k=known(p);return [int(x>=.5)if k[j]else'UNKNOWN'for j,x in enumerate(p)]
def score(p,y,v):return bool(np.all(known(p)&((p>=.5)==np.array(y))&np.array(v)))
def csvsave(path,rows):
 if not rows:return
 with Path(path).open('w',newline='',encoding='utf-8-sig')as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows({k:json.dumps(v,ensure_ascii=False)if isinstance(v,(dict,list))else v for k,v in r.items()}for r in rows)
def angle(a,b):return float(np.degrees(2*np.arccos(np.clip(abs(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))),0,1))))
def match(root,a,b,q):
 t=q['matching_tolerances'];rid=root['root_id'];dirs=[B/'episodes'/root['branches'][i]['episode_id']for i in [0,1]];inits=[dict(np.load(d/'_private/initial_state.npz'))for d in dirs];schema=read(dirs[0]/'_private/scene_layout.json');assert schema==read(dirs[1]/'_private/scene_layout.json');di=schema['drawer_index'];n=schema['nonobject_scene_length']
 initial=max(float(np.abs(inits[0]['robot_obs']-inits[1]['robot_obs']).max()),float(np.abs(np.delete(inits[0]['scene_obs']-inits[1]['scene_obs'],di)).max()));initial_ok=initial<=t['initial_non_drawer_max_abs'];deltas=[];checks=[]
 for k,(x,y)in enumerate(zip(a,b)):
  r0=np.array(x['robot_obs']);r1=np.array(y['robot_obs']);joint=np.abs((r0[7:14]-r1[7:14]+np.pi)%(2*np.pi)-np.pi);scene=np.abs(np.delete(np.array(x['scene_obs_private'][:n])-y['scene_obs_private'][:n],di));robot={'TCP_position_m':float(np.linalg.norm(r0[:3]-r1[:3])),'TCP_orientation_deg':angle(Rotation.from_euler('xyz',r0[3:6]).as_quat(),Rotation.from_euler('xyz',r1[3:6]).as_quat()),'arm_joint_deg':float(np.degrees(joint).max()),'gripper_width_m':float(abs(r0[6]-r1[6])),'gripper_command_equal':bool(r0[14]==r1[14]),'other_scene_max_abs':float(scene.max())};ro=robot['TCP_position_m']<=t['robot_TCP_position_m']and robot['TCP_orientation_deg']<=t['robot_TCP_orientation_deg']and robot['arm_joint_deg']<=t['robot_arm_joint_deg']and robot['gripper_width_m']<=t['gripper_width_m']and robot['gripper_command_equal']and robot['other_scene_max_abs']<=t['other_scene_joints_max_abs'];objs={}
  for c in COLORS:
   aa=x['objects_audit'][c];bb=y['objects_audit'][c];dp=float(np.linalg.norm(np.array(aa['position_m'])-bb['position_m']));da=angle(aa['quaternion_xyzw'],bb['quaternion_xyzw']);ok=dp<=t['object_position_m']and da<=t['object_orientation_deg'];objs[c]={'position_m':dp,'orientation_deg':da,'matched':ok};checks.append(ok)
  tx=x['targets'];ty=y['targets'];facts=all(tx['valid'])and all(ty['valid'])and tx['y'][:6]==ty['y'][:6]and not tx['y'][6]and ty['y'][6]and tx['y'][COLORS.index(root['other_color'])]and not tx['y'][COLORS.index(root['target_color'])];checks.extend([ro,facts]);deltas.append({'offset_controls':[-63,-42,-21,0][k],'closed_frame':x['frame'],'open_frame':y['frame'],'closed_drawer_m':x['drawer_m'],'open_drawer_m':y['drawer_m'],'robot':robot,'objects':objs,'robot_matched':bool(ro),'facts_matched_except_drawer':bool(facts)})
 return {'root_id':rid,'initial_non_drawer_max_abs':initial,'initial_matched':initial_ok,'history':deltas,'history_all_matched':all(checks),'strictly_matched':initial_ok and all(checks),'max_object_position_mm':max(o['position_m']*1000 for d in deltas for o in d['objects'].values()),'max_object_orientation_deg':max(o['orientation_deg']for d in deltas for o in d['objects'].values()),'max_robot_TCP_position_mm':max(d['robot']['TCP_position_m']*1000 for d in deltas),'max_robot_TCP_orientation_deg':max(d['robot']['TCP_orientation_deg']for d in deltas),'max_arm_joint_deg':max(d['robot']['arm_joint_deg']for d in deltas),'max_gripper_width_mm':max(d['robot']['gripper_width_m']*1000 for d in deltas),'max_other_scene_difference':max(d['robot']['other_scene_max_abs']for d in deltas)}
