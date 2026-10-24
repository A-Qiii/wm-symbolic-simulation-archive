from pathlib import Path
from collections import deque
import json,gzip,hashlib,sys,fcntl
import numpy as np
import pybullet as p
from omegaconf import OmegaConf
P=Path('/home/robot/research/projects/wm_symbolic');E=P/'experiments/state_predicates_held_down_20261014';B=P/'data/predicate_held_down_replay_20261014';sys.path.insert(0,str(P/'scripts'));sys.path.insert(0,str(P/'experiments/trajectory_boundary_mining_20261014/source'))
from wm_symbolic.env_adapter import CalvinAdapter,save_npz
from wm_symbolic.physical_oracle import PhysicalOracle
from core1200_collect import label,plain
from evaluate_physical_oracle import physics_guard
from mine_trajectories import state
C=['red','blue','pink']
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(plain(x),ensure_ascii=False,indent=2)+'\n')
def compare_image(ob,path,tag):
 with np.load(path)as z:
  q={'tag':tag,'source':str(path),'rgb':{},'mask':{}}
  for cam in ['static','gripper']:
   nr=ob['rgb_obs']['rgb_'+cam];nm=ob['segmentation'][cam];diff=np.abs(nr.astype(float)-z['rgb_'+cam].astype(float));q['rgb'][cam]={'bitwise_equal':bool(np.array_equal(nr,z['rgb_'+cam])),'mean_abs':float(diff.mean())};q['mask'][cam]={'bitwise_equal':bool(np.array_equal(nm,z['seg_'+cam]))}
 return q
def replay(source,protocol):
 root=source['root'];folder=B/'episodes'/source['episode_id'];folder.mkdir(exist_ok=False);(folder/'public').mkdir();original=Path(source['source_episode']);env=CalvinAdapter(P,root['environment'],root['seed'],tabletop=True,segmentation=True);checks=[];prefix=[];targettrace=[];segments=[]
 try:
  for obj in env.env.scene.movable_objects:
   v=root['layout'][obj.name];obj.initial_pos=np.array(v[:3]);obj.initial_orn=np.array(p.getQuaternionFromEuler(v[3:]))
  obs=env.reset();ctl=PhysicalOracle(env);OmegaConf.save(env.config,folder/'complete_config.yaml',resolve=True);np.savez_compressed(folder/'initial_state_private.npz',**env.env.get_state_obs());save(folder/'source_provenance.json',source)
  with physics_guard(env):
   for _ in range(24):obs=env.step([obs['robot_obs'][:3],obs['robot_obs'][3:6],1])
   for a in read(original/'setup_private.json')['controls']:obs=env.step([np.array(a['target_xyz']),np.array(a['target_quaternion']),int(a['gripper'])])
  rootstate=env.env.get_state_obs();np.savez_compressed(folder/'root_state_private.npz',**rootstate)
  with np.load(original/'root_reset_private.npz')as z:
   rooterrors={k:float(np.abs(rootstate[k]-z[k]).max())for k in ['robot_obs','scene_obs']};expected_previous={k:z[k].copy()for k in ['robot_obs','scene_obs']}
  for i,ob in enumerate(env.history):checks.append(compare_image(ob,original/f'root_observation_{i}.npz',f'root_history_{i}'))
  if max(rooterrors.values())>protocol['tolerances']['max_settled_scene_or_robot_vector_abs']:save(folder/'root_mismatch.json',rooterrors);raise RuntimeError('ROOT_REPLAY_OUTSIDE_FROZEN_TOLERANCE')
  target_path=Path(source['source_target'])
  for sp in source['prefix_segments']:
   sd=Path(sp);actions=read(sd/'controls.json');old=json.load(gzip.open(sd/'trace_private.json.gz','rt'));assert len(actions)==len(old);is_target=sd==target_path;ob=env.observe();s0=env.env.get_state_obs();initinfo=read(sd/'endpoint_info_private.json')['before'];initial_errors={k:float(np.abs(s0[k]-expected_previous[k]).max())for k in ['robot_obs','scene_obs']};checks.append(compare_image(ob,sd/'initial_public.npz',sd.name+':initial'));numeric={'robot_max_abs':0.,'robot_position_max_m':0.,'robot_angles_max_rad':0.,'gripper_width_max_m':0.,'scene_max_abs':0.,'object_pose_max_abs':0.};event_mismatches=[];sampled=[]
   def capture(obs,frame,action):
    st=env.env.get_state_obs();raw=[list(x)for x in p.getContactPoints(physicsClientId=env.env.cid)if x[9]>.01];objs={o.name:o.get_state().tolist()for o in env.env.scene.movable_objects};aud={}
    for c in C:
     uid=ctl.objects['block_'+c].uid;q=state(objs['block_'+c],raw,uid,ctl.robot.robot_uid,obs['robot_obs'],ctl.finger_families);q.update(position_m=objs['block_'+c][:3],object_uid_evaluator_only=uid,visible_pixels={cam:int((((m&((1<<24)-1))==uid)&(m>=0)).sum())for cam,m in obs['segmentation'].items()});aud[c]=q
    row={'frame':frame,'action':action,'robot_obs':obs['robot_obs'].tolist(),'scene_obs':st['scene_obs'].tolist(),'objects':objs,'contacts':raw,'objects_audit':aud}
    if is_target:row['labels']=label(env);targettrace.append(row);save_npz(folder/'public'/f'{frame:04d}.npz',obs)
    return row
   if is_target:capture(ob,0,{'phase':'initial_before_target','source':'continuousprefix,noreset'})
   with physics_guard(env):
    for i,a in enumerate(actions,1):
     ob=env.step([np.array(a['target_xyz']),np.array(a['target_quaternion']),int(a['gripper'])]);new=capture(ob,i,a);ref=old[i-1];dv=np.abs(np.array(new['robot_obs'])-ref['robot_obs']);numeric['robot_max_abs']=max(numeric['robot_max_abs'],float(dv.max()));numeric['robot_position_max_m']=max(numeric['robot_position_max_m'],float(np.linalg.norm(dv[:3])));numeric['robot_angles_max_rad']=max(numeric['robot_angles_max_rad'],float(dv[[3,4,5,7,8,9,10,11,12,13]].max()));numeric['gripper_width_max_m']=max(numeric['gripper_width_max_m'],float(dv[6]));numeric['scene_max_abs']=max(numeric['scene_max_abs'],float(np.abs(np.array(new['scene_obs'])-ref['scene_obs']).max()));numeric['object_pose_max_abs']=max(numeric['object_pose_max_abs'],max(float(np.abs(np.array(new['objects'][name])-v).max())for name,v in ref['objects'].items()))
     for c in C:
      uid=initinfo['scene_info']['movable_objects']['block_'+c]['uid'];expected=state(ref['objects']['block_'+c],ref['contacts'],uid,initinfo['robot_info']['uid'],ref['robot_obs'],ctl.finger_families)
      for field in ['held','bilateral','external']:
       if bool(new['objects_audit'][c][field])!=bool(expected[field]):event_mismatches.append({'frame':i,'colour':c,'field':field,'old':expected[field],'new':new['objects_audit'][c][field]})
     if not is_target:prefix.append({'segment':sd.name,**new})
     if len(actions)-3<=i<=len(actions):checks.append(compare_image(ob,sd/f'boundary_history_{i-(len(actions)-3)}.npz',sd.name+f':boundary_{i}'))
   final=env.observe();checks.append(compare_image(final,sd/'final_public.npz',sd.name+':final'))
   if old:expected_previous={'robot_obs':np.array(old[-1]['robot_obs']),'scene_obs':np.array(old[-1]['scene_obs'])}
   seg={'segment':str(sd),'actions':len(actions),'initial_state_errors':initial_errors,'numeric_errors':numeric,'event_mismatches':event_mismatches,'action_sha256':sha(sd/'controls.json'),'source_trace_sha256':sha(sd/'trace_private.json.gz')};segments.append(seg);save(folder/'progress_private.json',{'root_errors':rooterrors,'segments':segments})
   tol=protocol['tolerances'];ok=max(initial_errors.values())<=tol['max_settled_scene_or_robot_vector_abs']and numeric['robot_position_max_m']<=tol['max_TCP_or_object_position_m']and numeric['robot_angles_max_rad']<=tol['max_robot_or_object_angle_rad']and numeric['gripper_width_max_m']<=tol['max_gripper_width_m']and numeric['scene_max_abs']<=tol['max_settled_scene_or_robot_vector_abs']and not event_mismatches
   if not ok:raise RuntimeError('SEGMENT_REPLAY_OUTSIDE_FROZEN_TOLERANCE')
  with gzip.open(folder/'prefix_trace_private.json.gz','wt')as f:json.dump(plain(prefix),f)
  with gzip.open(folder/'target_trace_private.json.gz','wt')as f:json.dump(plain(targettrace),f)
  maxnum=max([*rooterrors.values()]+[v for s in segments for d in ['initial_state_errors','numeric_errors']for v in s[d].values()]);pixexact=all(v['bitwise_equal']for q in checks for typ in ['rgb','mask']for v in q[typ].values());exact=maxnum<=protocol['tolerances']['exact_numeric_max_abs']and pixexact and not any(s['event_mismatches']for s in segments);rgbmae=max(v['mean_abs']for q in checks for v in q['rgb'].values());within=maxnum<=protocol['tolerances']['max_settled_scene_or_robot_vector_abs']and rgbmae<=protocol['tolerances']['saved_RGB_mean_absolute_pixel_error_0_255']and pixexact;verification={'within_predefined_tolerance':bool(within),'exact_on_all_available_checks':bool(exact),'max_numeric_error':maxnum,'root_errors':rooterrors,'segments':segments,'image_checks':checks,'identity':'verified_replay_matches_available_original_records'if exact else'new_replay_not_original_missing_files','no_intermediate_state_reset':True};save(folder/'replay_verification.json',verification);assert within,'IMAGE_OR_STATE_REPLAY_OUTSIDE_FROZEN_TOLERANCE'
  anchor=source['anchor_frame'];inds=source['history_frame_indices'];c=source['target_color'];ts=targettrace[anchor-63:anchor+1];ro=np.array([t['robot_obs'][:3]for t in ts]);obj=np.array([t['objects_audit'][c]['position_m']for t in ts]);motion={'held_all64':all(t['objects_audit'][c]['held']for t in ts),'TCP_z_change_mm':float((ro[-1,2]-ro[0,2])*1000),'object_z_change_mm':float((obj[-1,2]-obj[0,2])*1000),'relative_drift_mm':float(np.linalg.norm((obj-ro)-(obj-ro)[0],axis=1).max()*1000)};assert motion['held_all64']and motion['TCP_z_change_mm']<=-20 and motion['object_z_change_mm']<=-20 and motion['relative_drift_mm']<=15
  lp=folder/'labels_private.json';save(lp,{'anchor_frame':anchor,'history_frame_indices':inds,'frames':[targettrace[i]for i in inds],'motion_audit_private':motion,'physical_definition_unchanged':True});visible=all(max(targettrace[i]['objects_audit'][c]['visible_pixels'].values())>=4 for i in inds);record={'sample_id':source['episode_id']+'__held_down','root_id':root['root_id'],'origin_root_id':root['root_id'],'episode_id':source['episode_id'],'trajectory_id':target_path.name,'environment':root['environment'],'split':root['split'],'dev_role':root['dev_role'],'split_status':source['split_status'],'target_color':c,'condition':'held_down','anchor_frame':anchor,'history_frame_indices':inds,'public_frames':[str(folder/'public'/f'{i:04d}.npz')for i in inds],'labels_private':str(lp),'target_visible_all4':visible,'new_independent_root':False};summary={'root_id':root['root_id'],'episode_id':source['episode_id'],'target_recorded_frames':len(targettrace),'prefix_actions':len(prefix),'setup_actions':len(read(original/'setup_private.json')['controls']),'settle_actions':24,'target_actions':len(targettrace)-1,'verification':verification,'motion':motion,'eligible_target':visible,'physics_guard_passed':True};save(folder/'summary.json',summary);return record,summary
 finally:env.close()
def main():
 protocol=read(E/'configs/protocol.json')
 for path,h in protocol['frozen_sha256'].items():assert sha(path)==h,path
 lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);manifest=read(B/'data_manifest.json');done={s['root_id']for s in manifest['episodes']}
 for source in read(B/'sources_manifest.json'):
  if source['root']['root_id']in done:continue
  record,summary=replay(source,protocol);manifest['records'].append(record);manifest['episodes'].append(summary);save(B/'data_manifest.json',manifest);print(json.dumps({'root':record['root_id'],'target_frames':summary['target_recorded_frames'],'exact':summary['verification']['exact_on_all_available_checks'],'target_visible':summary['eligible_target'],'motion':summary['motion']},ensure_ascii=False),flush=True)
 print('SIX_HELD_DOWN_REPLAYS_COMPLETE')
if __name__=='__main__':main()
