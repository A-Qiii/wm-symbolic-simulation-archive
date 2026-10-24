from pathlib import Path
import json,gzip,hashlib,sys,fcntl,time
import numpy as np
import pybullet as p
from omegaconf import OmegaConf
P=Path('/home/robot/research/projects/wm_symbolic');E=P/'experiments/state_predicates_countermotion_20261014';B=P/'data/predicate_countermotion_replay_20261014';sys.path.insert(0,str(P/'scripts'))
from wm_symbolic.env_adapter import CalvinAdapter,save_npz
from wm_symbolic.physical_oracle import PhysicalOracle
from core1200_collect import label,plain
from evaluate_physical_oracle import physics_guard
C=['red','blue','pink']
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(plain(x),ensure_ascii=False,indent=2)+'\n')
def ev(x):return[i for i in range(1,len(x))if x[i]!=x[i-1]]
def collect(root,protocol):
 folder=B/'episodes'/root['trajectory_id'];folder.mkdir(exist_ok=False);(folder/'public').mkdir();source=Path(root['source_trajectory']);old=json.load(gzip.open(source/'trace_private.json.gz','rt'));available={i:p for r in read(source/'samples_manifest.json')for i,p in zip(r['history_frame_indices'],r['public_frames'])};trace=[];checks=[];env=CalvinAdapter(P,'C',root['seed'],tabletop=True,segmentation=True)
 try:
  for obj in env.env.scene.movable_objects:
   v=root['layout'][obj.name];obj.initial_pos=np.array(v[:3]);obj.initial_orn=np.array(p.getQuaternionFromEuler(v[3:]))
  for door in env.env.scene.doors:
   if door.name=='base__drawer':door.initial_state=root['drawer_reset_m']
  obs=env.reset();ctl=PhysicalOracle(env);OmegaConf.save(env.config,folder/'complete_config.yaml',resolve=True);np.savez_compressed(folder/'initial_state_private.npz',**env.env.get_state_obs());save(folder/'root_provenance.json',root)
  def record(ob,action):
   i=len(trace);lab=label(env);info=env.env.get_info();rb=info['robot_info'];objects={};raw={}
   for c in C:
    obj=ctl.objects['block_'+c];cs=[cc for cc in p.getContactPoints(bodyA=obj.uid,physicsClientId=env.env.cid)if cc[9]>.01];raw['block_'+c]=[list(z)for z in cs];rc=[z for z in cs if z[2]==ctl.robot.robot_uid];links={z[4]for z in rc};bi=all(links&family for family in ctl.finger_families);ex=[z for z in cs if z[2]!=ctl.robot.robot_uid];pos=ctl.xyz('block_'+c);pixels={cam:int((((mask&((1<<24)-1))==obj.uid)&(mask>=0)).sum())for cam,mask in ob['segmentation'].items()};objects[c]={'position_m':pos.tolist(),'bilateral_finger_force':bool(bi),'external_support_contacts':len(ex),'robot_force_contacts':len(rc),'held_contact_audit':bool(bi and not ex),'tcp_distance_m':float(np.linalg.norm(pos-np.array(rb['tcp_pos']))),'visible_pixels':pixels,'object_uid_evaluator_only':obj.uid};assert bool(lab['holding_derived'][C.index(c)])==objects[c]['held_contact_audit']
   trace.append({'frame':i,'action':action,'labels':lab,'robot_obs':ob['robot_obs'].tolist(),'tcp_pos':list(rb['tcp_pos']),'gripper_command':int(rb['gripper_action']),'gripper_width_m':float(rb['gripper_opening_width']),'objects_audit':objects,'contacts_private':raw,'scene_obs_private':env.env.scene.get_obs().tolist()});save_npz(folder/'public'/f'{i:04d}.npz',ob)
   if i in available:
    with np.load(available[i])as z:
     row={'frame':i,'source':available[i],'rgb':{},'mask':{}};uid=objects[root['target_color']]['object_uid_evaluator_only']
     for cam in ['static','gripper']:
      nr=ob['rgb_obs']['rgb_'+cam];diff=np.abs(nr.astype(float)-z['rgb_'+cam].astype(float));row['rgb'][cam]={'bitwise_equal':bool(np.array_equal(nr,z['rgb_'+cam])),'mean_abs':float(diff.mean())};nm=ob['segmentation'][cam];om=z['seg_'+cam];a=nm==uid;b=om==uid;union=int((a|b).sum());row['mask'][cam]={'bitwise_equal':bool(np.array_equal(nm,om)),'target_IoU':float((a&b).sum()/union)if union else 1.}
    checks.append(row)
   if i==24:
    state=env.env.get_state_obs();np.savez_compressed(folder/'settled_state_private.npz',**state)
    with np.load(source/'settled_root_private.npz')as z:save(folder/'settled_comparison.json',{k:float(np.abs(state[k]-z[k]).max())for k in ['robot_obs','scene_obs']})
  record(obs,old[0]['action'])
  with physics_guard(env):
   for row in old[1:]:
    a=row['action'];record(env.step([np.array(a['target_xyz']),np.array(a['target_quaternion']),int(a['gripper'])]),a)
  with gzip.open(folder/'trace_private.json.gz','wt')as f:json.dump(plain(trace),f)
  assert len(trace)==len(old);nr=np.array([r['robot_obs']for r in trace]);oo=np.array([r['robot_obs']for r in old]);dv=np.abs(nr-oo);pos=max(float(np.linalg.norm(np.array(n['objects_audit'][c]['position_m'])-o['objects_audit'][c]['position_m']))for n,o in zip(trace,old)for c in C);events=[]
  for c in C:
   for k in ['held_contact_audit','bilateral_finger_force','external_support_contacts']:
    aa=[bool(t['objects_audit'][c][k])for t in trace];bb=[bool(t['objects_audit'][c][k])for t in old];ae=ev(aa);be=ev(bb);shift=max([abs(a-b)for a,b in zip(ae,be)]or[0])if len(ae)==len(be)and aa[0]==bb[0]else 999;events.append({'colour':c,'fact':k,'old':be,'new':ae,'max_shift':shift,'same_values':aa==bb})
  settled=read(folder/'settled_comparison.json');numeric={'tcp_position_m':float(np.linalg.norm(nr[:,:3]-oo[:,:3],axis=1).max()),'object_position_m':pos,'robot_angles_rad':float(dv[:,[3,4,5,7,8,9,10,11,12,13]].max()),'aperture_m':float(dv[:,6].max()),'all_robot_abs':float(dv.max()),'settled_max_abs':max(settled.values())};tol=protocol['tolerances'];within=numeric['tcp_position_m']<=tol['max_TCP_or_object_position_m']and pos<=tol['max_TCP_or_object_position_m']and numeric['robot_angles_rad']<=tol['max_robot_or_object_angle_rad']and numeric['aperture_m']<=tol['max_gripper_width_m']and numeric['settled_max_abs']<=tol['max_settled_scene_or_robot_vector_abs']and max(r['max_shift']for r in events)<=tol['event_transition_max_shift_controls']and max(x['mean_abs']for q in checks for x in q['rgb'].values())<=tol['saved_RGB_mean_absolute_pixel_error_0_255']and min(x['target_IoU']for q in checks for x in q['mask'].values())>=tol['saved_target_mask_IoU_min'];exact=max(numeric.values())<=tol['exact_numeric_max_abs']and all(q['same_values']for q in events)and all(z['bitwise_equal']for q in checks for typ in ['rgb','mask']for z in q[typ].values());verification={'within_predefined_tolerance':bool(within),'exact_on_all_available_checks':bool(exact),'numeric_errors':numeric,'events':events,'saved_frame_checks':checks,'original_actions_identical':[q['action']for q in trace]==[q['action']for q in old],'identity':'verified_replay_matches_available_original_records'if exact else'new_replay_not_original_missing_files'};save(folder/'replay_verification.json',verification);assert within,'REPLAY_OUTSIDE_FROZEN_TOLERANCE'
  anchor=root['anchor_frame'];inds=root['history_frame_indices'];lp=folder/'labels_private.json';save(lp,{'anchor_frame':anchor,'history_frame_indices':inds,'frames':[trace[i]for i in inds],'current_label':trace[anchor]['labels'],'physical_definition_unchanged':True});visible=all(max(trace[i]['objects_audit'][root['target_color']]['visible_pixels'].values())>=4 for i in inds);record={'sample_id':root['trajectory_id']+'__upward_empty','root_id':root['root_id'],'origin_root_id':root['root_id'],'trajectory_id':root['trajectory_id'],'environment':'C','split':'dev','condition':'upward_empty_closed','anchor_frame':anchor,'history_frame_indices':inds,'target_color':root['target_color'],'public_frames':[str(folder/'public'/f'{i:04d}.npz')for i in inds],'labels_private':str(lp),'target_visible_all4':visible,'eligible':visible,'new_independent_root':False,'replay_identity':verification['identity']};summary={'root_id':root['root_id'],'trajectory_id':root['trajectory_id'],'frames':len(trace),'actions':len(trace)-1,'source_trace_sha256':sha(source/'trace_private.json.gz'),'verification':verification,'physics_guard_passed':True,'anchor_fixed_before_replay':anchor,'eligible':visible,'target_visibility_pixels':[trace[i]['objects_audit'][root['target_color']]['visible_pixels']for i in inds]};save(folder/'summary.json',summary);return record,summary
 finally:env.close()
def main():
 protocol=read(E/'configs/protocol.json')
 for p,h in protocol['frozen_sha256'].items():assert sha(p)==h,p
 lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);m=read(B/'data_manifest.json');done={r['root_id']for r in m['trajectories']}
 for root in read(B/'roots_manifest.json'):
  if root['root_id']in done:continue
  record,summary=collect(root,protocol);m['records'].append(record);m['trajectories'].append(summary);save(B/'data_manifest.json',m);print(json.dumps({'root':root['root_id'],'frames':summary['frames'],'exact':summary['verification']['exact_on_all_available_checks'],'eligible':summary['eligible']},ensure_ascii=False),flush=True)
 print('COUNTERMOTION_REPLAY_COMPLETE')
if __name__=='__main__':main()
