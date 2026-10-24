"""Offline labels only. Never imported by a world-model or planner forward pass."""
import numpy as np
COLORS=('red','blue','pink')
LOCATIONS=('table_left','table_middle','table_right','drawer','slider_left','slider_right','held','other')
SPEC={'version':'geometry_v1','units':'metres and radians','table_bands':'thirds of each native scene table sampling bounds',
 'boundary_margin_m':0.005,'drawer_closed_max':0.025,'drawer_open_min':0.16,
 'slider_right_max':0.055,'slider_left_min':0.22,'moved_displacement_m':0.02,
 'held_max_tcp_distance_m':0.075,'held_contacts':'closing gripper, contact with a finger, no supporting table/part contact, near TCP',
 'unresolved':'Label validity mask is false near boundaries or ambiguous grasps; other is a physical category, not UNKNOWN',
 'source':'Native scene calibration, full control-step evaluator contacts/positions; never model inputs',
 'frozen_before':'Any world-model training or test observation'}

def other_contact(c,uid):
 if c[1]==uid:return c[2],c[4]
 if c[2]==uid:return c[1],c[3]
 return None,None

def labels_at(info,scene_cfg):
 scene=info['scene_info'];robot=info['robot_info'];table=scene['fixed_objects']['table'];rid=robot['uid']
 bounds=np.asarray(scene_cfg['surfaces']['table']);x1,x2=np.linspace(bounds[0,0],bounds[1,0],4)[1:3]
 left=np.asarray(scene_cfg['surfaces']['slider_left']);right=np.asarray(scene_cfg['surfaces']['slider_right'])
 slider_mid=(left[1,0]+right[0,0])/2
 cats=[];valid=[];continuous={};grip_candidates=[]
 for color in COLORS:
  o=scene['movable_objects']['block_'+color];pos=np.asarray(o['current_pos']);uid=o['uid'];contacts=[other_contact(c,uid) for c in o['contacts'] if c[9]>1e-4]
  support={link for body,link in contacts if body==table['uid']}
  fingers={link for body,link in contacts if body==rid and link in [9,10]}
  distance=float(np.linalg.norm(pos-np.asarray(robot['tcp_pos'])));held=bool(fingers) and not support and robot['gripper_action']<0 and distance<SPEC['held_max_tcp_distance_m']
  ok=True
  if held:label=6;grip_candidates.append(len(cats))
  elif table['links']['drawer_link'] in support:label=3
  elif table['links']['plank_link'] in support:
   label=4 if pos[0]<slider_mid else 5;ok=abs(pos[0]-slider_mid)>SPEC['boundary_margin_m']
  elif table['links']['base_link'] in support:
   label=0 if pos[0]<x1 else (1 if pos[0]<x2 else 2);ok=min(abs(pos[0]-x1),abs(pos[0]-x2))>SPEC['boundary_margin_m']
  else:
   label=7
   if fingers or distance<SPEC['held_max_tcp_distance_m'] and robot['gripper_action']<0:ok=False
  cats.append(int(label));valid.append(bool(ok))
  continuous[color]={'position_m':pos.tolist(),'finger_contacts':sorted(fingers),'support_links':sorted(support),'tcp_distance_m':distance}
 if len(grip_candidates)>1:
  for j in grip_candidates:valid[j]=False
 drawer=scene['doors']['base__drawer']['current_state'];slide=scene['doors']['base__slide']['current_state']
 for value,lo,hi in [(drawer,SPEC['drawer_closed_max'],SPEC['drawer_open_min']),(slide,SPEC['slider_right_max'],SPEC['slider_left_min'])]:
  cats.append(0 if value<lo else (2 if value>hi else 1));valid.append(min(abs(value-lo),abs(value-hi))>SPEC['boundary_margin_m'])
 binary=[int(scene['lights'][name]['logical_state']) for name in ['lightbulb','led']]
 return {'categorical':cats,'categorical_valid':valid,'binary':binary,'binary_valid':[True,True],
  'continuous':{'objects':continuous,'drawer_m':drawer,'slider_m':slide},
  'holding_derived':[bool(ok and val==6) for val,ok in zip(cats[:3],valid[:3])],
  'hand_empty_derived':None if not all(valid[:3]) else not any(v==6 for v in cats[:3])}

def annotate_trace(infos,scene_cfg,success):
 first=infos[0];start=np.array([first['scene_info']['movable_objects']['block_'+c]['current_pos'] for c in COLORS])
 max_displacement=np.zeros(3);first_event=[None]*3
 for t,info in enumerate(infos):
  pos=np.array([info['scene_info']['movable_objects']['block_'+c]['current_pos'] for c in COLORS])
  d=np.linalg.norm(pos-start,axis=-1);max_displacement=np.maximum(d,max_displacement)
  for j in range(3):
   if first_event[j] is None and d[j]>SPEC['moved_displacement_m']:first_event[j]=t
 return {'schema':'labels_v1','before':labels_at(first,scene_cfg),'after':labels_at(infos[-1],scene_cfg),
  'success':bool(success),'events':(max_displacement>SPEC['moved_displacement_m']).astype(int).tolist(),'events_valid':[True]*3,
  'event_max_displacement_m':max_displacement.tolist(),'event_first_control_step':first_event,
  'control_samples_including_start':len(infos),'specification':SPEC,'role':'offline_training_and_scoring_only'}
