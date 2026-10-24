from pathlib import Path
import sys,json,hashlib,time,datetime,fcntl,gzip,argparse
import numpy as np
import pybullet as p
import imageio.v2 as imageio
P=Path('/home/robot/research/projects/wm_symbolic');sys.path.insert(0,str(P/'scripts'))
from wm_symbolic.env_adapter import CalvinAdapter,save_npz
from wm_symbolic.physical_oracle import PhysicalOracle,ExecutionFailure
from core1200_collect import label,plain
from evaluate_physical_oracle import physics_guard
B=P/'data/predicate_joint_acceptance_20261014';R=P/'experiments/state_predicates_joint_20261014';C=['red','blue','pink']
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,obj):
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(plain(obj),ensure_ascii=False,indent=2));tmp.replace(path)
def collect(root,branch):
 eid=root['root_id']+'__'+branch;out=B/'episodes'/eid;out.mkdir(exist_ok=False)
 env=CalvinAdapter(P,root['environment'],root['seed'],tabletop=True,segmentation=True);trace=[];obs_all=[];writers={};ctl=None;result=None;issues=[];motion={};start=time.monotonic()
 try:
  for obj in env.env.scene.movable_objects:
   val=root['layout'][obj.name];obj.initial_pos=np.array(val[:3]);obj.initial_orn=np.array(p.getQuaternionFromEuler(val[3:]))
  for door in env.env.scene.doors:
   if door.name=='base__drawer':door.initial_state=root['drawer_reset_m']
  obs=env.reset();ctl=PhysicalOracle(env);target='block_'+root['target_color'];j=C.index(root['target_color'])
  writers={cam:imageio.get_writer(str(out/(cam+'.mp4')),fps=30,macro_block_size=1,codec='libx264',quality=8) for cam in obs['rgb_obs']}
  def record(obs,act):
   labels=label(env);infos=env.env.get_info();rb=infos['robot_info'];ob={};raw_contacts={}
   for color in C:
    name='block_'+color;obj=ctl.objects[name];contacts=[c for c in p.getContactPoints(bodyA=obj.uid,physicsClientId=env.env.cid) if c[9]>.01];raw_contacts[name]=[list(c) for c in contacts];rcontacts=[c for c in contacts if c[2]==ctl.robot.robot_uid];fingerlinks={c[4] for c in rcontacts};bilateral=all(fingerlinks&family for family in ctl.finger_families);external=[c for c in contacts if c[2]!=ctl.robot.robot_uid];pos=ctl.xyz(name)
    vis={cam:int((((seg&((1<<24)-1))==obj.uid)&(seg>=0)).sum()) for cam,seg in obs['segmentation'].items()}
    ob[color]={'position_m':pos.tolist(),'tcp_distance_m':float(np.linalg.norm(pos-np.array(rb['tcp_pos']))),'bilateral_finger_force':bool(bilateral),'external_support_contacts':len(external),'held_contact_audit':bool(bilateral and not external),'visible_pixels':vis,'object_uid_evaluator_only':obj.uid}
    if bool(labels['holding_derived'][C.index(color)])!=ob[color]['held_contact_audit']:issues.append({'frame':len(trace),'color':color,'reason':'existing_label_contact_disagreement'})
   dv={cam:int((((seg&((1<<24)-1))==ctl.table.uid)&((seg>>24)-1==ctl.links['drawer_link'])).sum()) for cam,seg in obs['segmentation'].items()}
   trace.append({'frame':len(trace),'action':act,'labels':labels,'robot_obs':obs['robot_obs'].tolist(),'tcp_pos':list(rb['tcp_pos']),'gripper_command':int(rb['gripper_action']),'gripper_width_m':float(rb['gripper_opening_width']),'objects_audit':ob,'drawer_visible_pixels':dv,'drawer_threshold_distances_m':{'closed':float(labels['continuous']['drawer_m']-.025),'open':float(labels['continuous']['drawer_m']-.16)},'contacts_private':raw_contacts})
   obs_all.append(obs)
   for cam,w in writers.items():w.append_data(obs['rgb_obs'][cam])
  ctl.recorder=record;record(obs,{'phase':'reset_snapshot','source':'initial reset only'})
  ctl.phase='reset_settle'
  with physics_guard(env):ctl.hold(24,1)
  state=env.env.get_state_obs();np.savez_compressed(out/'settled_root_private.npz',**state)
  rootdir=B/'roots'/root['root_id'];rootdir.mkdir(exist_ok=True)
  if branch=='grasp':np.savez_compressed(rootdir/'reference_settled_root_private.npz',**state);replay={'scene_max_abs_diff':0.,'robot_max_abs_diff':0.}
  else:
   with np.load(rootdir/'reference_settled_root_private.npz') as z:replay={'scene_max_abs_diff':float(np.abs(z['scene_obs']-state['scene_obs']).max()),'robot_max_abs_diff':float(np.abs(z['robot_obs']-state['robot_obs']).max())}
  if max(replay.values())>1e-5:issues.append({'reason':'root_replay_mismatch',**replay})
  save(out/'reset_provenance.json',{'root':root,'reset_mutations':'Initial block poses and drawer state only, prior to physics guard','settle_controls':24,'root_replay':replay,'held_teleport':False})
  if branch=='grasp':
   with physics_guard(env):result=ctl.run('lift_'+root['target_color']+'_block_table')
  elif branch=='place':
   with physics_guard(env):
    lift_result=ctl.run('lift_'+root['target_color']+'_block_table')
    result=ctl.run('place_in_drawer') if lift_result['status']=='success' else {'status':'skipped_place_after_lift_failure','lift_result':lift_result}
   result['preceding_lift_result']=lift_result
  else:
   ctl.steps=0;ctl.events=[];orn=ctl.block_orn(target);orn[2]+=root['near_probe_yaw_offset_rad'];origin=ctl.xyz(target).copy();near=origin+root['near_probe_translation_m']
   try:
    with physics_guard(env):
     ctl.safe_height(orn,-1);ctl.phase='far_closed_hold';ctl.hold(20,-1)
     ctl.move([near[0],near[1],.62],orn,-1,'above_miss_probe')
     ctl.move(near,orn,-1,'descend_miss_probe',speed=.0025,settle=12);ctl.phase='near_closed_hold';ctl.hold(12,-1)
     motion['before_frame']=len(trace)-1
     ctl.move(near+root['follow_probe_translation_m'],orn,-1,'near_empty_follow_probe',speed=.0025,settle=12);ctl.phase='near_motion_hold';ctl.hold(12,-1)
     motion['after_frame']=len(trace)-1
    result={'status':'completed_probe','controls':ctl.steps,'events':ctl.events,'skill_modified':False}
   except ExecutionFailure as ex:result={'status':'failed_probe','controls':ctl.steps,'reason':str(ex),'events':ctl.events}
  # Independent temporal audit; it never changes original categorical labels.
  if branch in ['grasp','place']:
   candidates=[i for i,t in enumerate(trace) if t['action']['phase']=='lift_named_block' and t['objects_audit'][root['target_color']]['held_contact_audit']]
   best=None
   for starti in candidates:
    end=starti
    while end+1<len(trace) and trace[end+1]['action']['phase']=='lift_named_block' and trace[end+1]['objects_audit'][root['target_color']]['held_contact_audit']:end+=1
    if end-starti<3:continue
    tcp=np.array([t['tcp_pos'] for t in trace[starti:end+1]]);obj=np.array([t['objects_audit'][root['target_color']]['position_m'] for t in trace[starti:end+1]]);delta=float(np.linalg.norm(tcp[-1]-tcp[0]));relative=float(np.linalg.norm((obj-tcp)-(obj-tcp)[0],axis=1).max());objdelta=float(np.linalg.norm(obj[-1]-obj[0]));item={'start_frame':starti,'end_frame':end,'tcp_displacement_m':delta,'object_displacement_m':objdelta,'relative_drift_max_m':relative,'verified':delta>=.02 and objdelta>=.02 and relative<=.015}
    if best is None or delta>best['tcp_displacement_m']:best=item
   motion={'held_follow_audit':best,'verified':bool(best and best['verified'])}
   endpoint_held=trace[-1]['objects_audit'][root['target_color']]['held_contact_audit']
   if endpoint_held and best and best['tcp_displacement_m']>=.02 and not best['verified']:issues.append({'reason':'held_contacts_but_failed_motion_agreement','motion':best})
  elif 'after_frame' in motion:
   ts=trace[motion['before_frame']:motion['after_frame']+1];tcp=np.array([t['tcp_pos'] for t in ts]);obj=np.array([t['objects_audit'][root['target_color']]['position_m'] for t in ts]);td=float(np.linalg.norm(tcp[-1]-tcp[0]));od=float(np.linalg.norm(obj-obj[0],axis=1).max());nonheld=all(not t['objects_audit'][root['target_color']]['held_contact_audit'] for t in ts);supported=all(t['objects_audit'][root['target_color']]['external_support_contacts']>0 for t in ts)
   motion.update(tcp_displacement_m=td,object_max_displacement_m=od,continuous_not_held=nonheld,persistent_external_support=supported,verified=td>=.008 and od<=.003 and nonheld and supported)
   if nonheld and td>=.008 and od>.012:issues.append({'reason':'negative_contact_label_but_object_follows_probe','motion':motion})
  else:motion['verified']=False
  records=[];gaps=[]
  def choose(name,fn,need_motion=False):
   eligible=[i for i in range(3,len(trace)) if all(fn(t) for t in trace[i-3:i+1])]
   if not eligible:gaps.append({'condition':name,'reason':'no_stable_four_frame_window'});return
   i=eligible[-1]
   if need_motion and not motion.get('verified'):gaps.append({'condition':name,'reason':'motion_not_verified','candidate_frame':i});return
   sd=out/'samples'/name;sd.mkdir(parents=True);frames=[];hashes=[]
   for k,t in enumerate(range(i-3,i+1)):
    file=sd/f'public_{k}.npz';save_npz(file,obs_all[t]);frames.append(str(file));h=hashlib.sha256(obs_all[t]['rgb_obs']['rgb_static'].tobytes()+obs_all[t]['rgb_obs']['rgb_gripper'].tobytes()).hexdigest();hashes.append(h)
   labfile=sd/'labels_private.json';save(labfile,{'frames':trace[i-3:i+1],'anchor_frame':i,'target_color':root['target_color'],'original_label_definition_unchanged':True,'motion_audit':motion,'model_must_not_read':'all contents of this file'})
   row={'sample_id':eid+'__'+name,'root_id':root['root_id'],'trajectory_id':eid,'environment':root['environment'],'split':root['split'],'condition':name,'target_color':root['target_color'],'planned_drawer':root['drawer_setting'],'anchor_frame':i,'history_frame_indices':list(range(i-3,i+1)),'public_frames':frames,'labels_private':str(labfile),'frame_rgb_sha256':hashes,'window_rgb_sha256':hashlib.sha256(''.join(hashes).encode()).hexdigest(),'actual_held':trace[i]['objects_audit'][root['target_color']]['held_contact_audit'],'actual_drawer_category':trace[i]['labels']['categorical'][3],'target_visible':any(x>=4 for x in trace[i]['objects_audit'][root['target_color']]['visible_pixels'].values()),'motion_verified':motion.get('verified',False)};records.append(row)
  color=root['target_color']
  def neg(t):return not t['objects_audit'][color]['held_contact_audit'] and t['labels']['categorical_valid'][j] and t['objects_audit'][color]['external_support_contacts']>0 and any(x>=4 for x in t['objects_audit'][color]['visible_pixels'].values())
  def nearneg(t):return neg(t) and t['objects_audit'][color]['tcp_distance_m']<.075
  if branch=='grasp':
   choose('far_open_negative',lambda t:neg(t) and t['action']['phase']=='reset_settle' and t['objects_audit'][color]['tcp_distance_m']>.12)
   choose('near_open_negative',lambda t:nearneg(t) and t['action']['phase']=='descend_named_block' and t['gripper_command']==1)
   choose('contact_supported_close_command_negative',lambda t:nearneg(t) and t['action']['phase']=='descend_named_block' and t['gripper_command']==-1)
   choose('held_positive',lambda t:t['objects_audit'][color]['held_contact_audit'] and t['action']['phase']=='verify_stable_completion',True)
  elif branch=='place':
   choose('in_positive',lambda t:t['labels']['categorical'][j]==3 and t['labels']['categorical_valid'][j] and not t['objects_audit'][color]['held_contact_audit'] and t['action']['phase']=='verify_stable_completion')
  else:
   choose('far_closed_negative',lambda t:neg(t) and t['action']['phase']=='far_closed_hold' and t['objects_audit'][color]['tcp_distance_m']>.12 and t['gripper_width_m']<=.015)
   choose('near_closed_negative',lambda t:nearneg(t) and t['action']['phase']=='near_closed_hold' and t['gripper_width_m']<=.015,True)
   choose('near_closed_motion_negative',lambda t:nearneg(t) and t['action']['phase']=='near_motion_hold' and t['gripper_width_m']<=.015,True)
  summary={'trajectory_id':eid,'root_id':root['root_id'],'split':root['split'],'branch':branch,'target_color':color,'drawer_planned':root['drawer_setting'],'source':'unchanged fixed skill or skill chain' if branch in ['grasp','place'] else 'fixed measurement sequence via unchanged motor primitives','controller_result':result,'controller_sha256':sha(P/'src/wm_symbolic/physical_oracle.py'),'frames':len(trace),'controls':len(trace)-1,'selected_windows':len(records),'gaps':gaps,'label_issues':issues,'root_replay':replay,'physics_guard_passed':True,'motion_audit':motion,'elapsed_seconds':time.monotonic()-start,'public_video_files':[str(out/(cam+'.mp4')) for cam in writers]}
  with gzip.open(out/'trace_private.json.gz','wt') as f:json.dump(plain(trace),f)
  save(out/'samples_manifest.json',records);save(out/'summary.json',summary)
  return records,summary
 finally:
  for w in writers.values():w.close()
  env.close()
