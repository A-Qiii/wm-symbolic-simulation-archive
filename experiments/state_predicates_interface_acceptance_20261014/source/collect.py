from pathlib import Path
import json,gzip,hashlib,sys,time,fcntl,traceback
import numpy as np
import pybullet as p
from omegaconf import OmegaConf
P=Path('/home/robot/research/projects/wm_symbolic');E=P/'experiments/state_predicates_interface_acceptance_20261014';B=P/'data/predicate_interface_acceptance_20261014';sys.path.insert(0,str(P/'scripts'))
from wm_symbolic.env_adapter import CalvinAdapter,save_npz
from wm_symbolic.physical_oracle import PhysicalOracle
from core1200_collect import label,plain
from evaluate_physical_oracle import physics_guard
C=['red','blue','pink']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(plain(x),ensure_ascii=False,indent=2)+'\n')
def collect(root,protocol):
 folder=B/'episodes'/root['root_id'];folder.mkdir(exist_ok=False);(folder/'public').mkdir();save(folder/'root_provenance.json',root);trace=[];attempts=[];segment='reset';windows={};failure=None;env=None;start=time.monotonic()
 try:
  env=CalvinAdapter(P,'D',root['seed'],tabletop=True,segmentation=True)
  for obj in env.env.scene.movable_objects:
   val=root['layout'][obj.name];obj.initial_pos=np.array(val[:3]);obj.initial_orn=np.array(p.getQuaternionFromEuler(val[3:]))
  for door in env.env.scene.doors:
   if door.name=='base__drawer':door.initial_state=root['drawer_reset_m']
  ob=env.reset();auditctl=PhysicalOracle(env);OmegaConf.save(env.config,folder/'complete_config.yaml',resolve=True);np.savez_compressed(folder/'initial_state_private.npz',**env.env.get_state_obs())
  def record(obs,action):
   lab=label(env);objects={};contacts={}
   for c in C:
    obj=auditctl.objects['block_'+c];cs=[q for q in p.getContactPoints(bodyA=obj.uid,physicsClientId=env.env.cid)if q[9]>.01];contacts[c]=[list(q)for q in cs];robot=[q for q in cs if q[2]==auditctl.robot.robot_uid];links={q[4]for q in robot};bi=all(links&family for family in auditctl.finger_families);ext=[q for q in cs if q[2]!=auditctl.robot.robot_uid];held=bool(bi and not ext);assert held==bool(lab['holding_derived'][C.index(c)])
    objects[c]={'position_m':auditctl.xyz('block_'+c).tolist(),'held':held,'bilateral':bool(bi),'external_support_contacts':len(ext),'robot_force_contacts':len(robot),'tcp_distance_m':float(np.linalg.norm(auditctl.xyz('block_'+c)-obs['robot_obs'][:3])),'object_uid_evaluator_only':obj.uid,'visible_pixels':{cam:int((((m&((1<<24)-1))==obj.uid)&(m>=0)).sum())for cam,m in obs['segmentation'].items()}}
   row={'frame':len(trace),'segment':segment,'action':action,'robot_obs':obs['robot_obs'].tolist(),'scene_obs_private':env.env.scene.get_obs().tolist(),'labels':lab,'objects_audit':objects,'contacts_private':contacts};save_npz(folder/'public'/f'{len(trace):04d}.npz',obs);trace.append(row)
  record(ob,{'phase':'initial_reset_only'})
  def hold(name,grip):
   nonlocal segment
   segment=name;ctl=PhysicalOracle(env,record);ctl.phase=name;s=len(trace)
   with physics_guard(env):ctl.hold(96,grip)
   r={'request':name,'kind':'unchanged_motor_hold','status':'executed','controls':len(trace)-s,'first_frame':s,'last_frame':len(trace)-1,'gripper':grip};attempts.append(r);save(folder/'attempts_private.json',attempts);return r
  def skill(name):
   nonlocal segment
   segment=name;ctl=PhysicalOracle(env,record);s=len(trace)
   with physics_guard(env):r=ctl.run(name)
   r.update(request=name,kind='existing_frozen_skill',first_frame=s,last_frame=len(trace)-1);attempts.append(r);save(folder/'attempts_private.json',attempts);return r
  hold('baseline_hold_96',1);windows['baseline']=len(trace)-1
  lift=skill('lift_'+root['target_color']+'_block_table')
  rows=[q for q in trace if q['segment']==lift['request']]
  nearopen=[q['frame']for q in rows if q['action']['phase']=='descend_named_block'and q['action']['gripper']==1];nearclosed=[q['frame']for q in rows if q['action']['phase']=='descend_named_block'and q['action']['gripper']==-1]
  if nearopen:windows['near_open']=nearopen[-1]
  if nearclosed:windows['near_closed']=nearclosed[0]
  if rows:windows['held_after_lift']=rows[-1]['frame']
  if lift['status']=='success':
   hold('held_hold_96',-1);windows['held_stationary']=len(trace)-1
   if root['drawer_setting']=='open':
    placed=skill('place_in_drawer');lower=[q['frame']for q in trace if q['segment']=='place_in_drawer'and q['action']['phase']=='lower_into_drawer']
    if len(lower)>=64 and lower[63]-lower[0]==63:windows['held_downward']=lower[63]
    if placed['status']=='success':
     hold('in_open_hold_96',1);windows['in_open']=len(trace)-1;closed=skill('close_drawer')
     if closed['status']=='success':hold('in_closed_hold_96',1);windows['in_closed']=len(trace)-1
  np.savez_compressed(folder/'final_state_private.npz',**env.env.get_state_obs())
 except Exception as ex:
  failure={'type':type(ex).__name__,'message':str(ex),'traceback':traceback.format_exc(),'last_frame':len(trace)-1};save(folder/'collection_error.json',failure)
 finally:
  if env is not None:env.close()
  with gzip.open(folder/'trace_private.json.gz','wt')as f:json.dump(plain(trace),f)
 target=root['target_color'];j=C.index(target)
 for i in range(63,len(trace)):
  inds=[i-63,i-42,i-21,i];pix=[trace[k]['objects_audit'][target]['visible_pixels']for k in inds];cur=pix[-1];one=any(cur[cam]<4 and max(q[cam]for q in pix)>=20 and cur[other]>=4 for cam,other in [('static','gripper'),('gripper','static')]);both=max(cur.values())<4 and max(v for q in pix for v in q.values())>=20
  if one:windows.setdefault('natural_one_camera_occlusion',i)
  if one and trace[i]['objects_audit'][target]['held']:windows.setdefault('natural_held_occlusion',i)
  if both:windows.setdefault('natural_both_camera_occlusion',i)
 planned=['baseline','near_open','near_closed','held_after_lift','held_stationary']+(['held_downward','in_open','in_closed']if root['drawer_setting']=='open'else[])+['natural_one_camera_occlusion','natural_held_occlusion','natural_both_camera_occlusion'];records=[];missing=[]
 for condition in planned:
  anchor=windows.get(condition)
  if anchor is None or anchor<63:
   missing.append({'condition':condition,'reason':'natural_event_not_observed'if condition.startswith('natural')else'insufficient_history_or_unreached_control_phase','anchor':anchor});continue
  inds=[anchor-63,anchor-42,anchor-21,anchor];q=trace[anchor];o=q['objects_audit'][target];win=trace[anchor-63:anchor+1];tcp=np.array([z['robot_obs'][:3]for z in win]);obj=np.array([z['objects_audit'][target]['position_m']for z in win]);dz=np.diff(tcp[:,2])*1000;oz=np.diff(obj[:,2])*1000;motion={'held_all64':all(z['objects_audit'][target]['held']for z in win),'TCP_displacement_from_start_max_mm':float(np.linalg.norm(tcp-tcp[0],axis=1).max()*1000),'object_displacement_from_start_max_mm':float(np.linalg.norm(obj-obj[0],axis=1).max()*1000),'TCP_net_dz_mm':float((tcp[-1,2]-tcp[0,2])*1000),'object_net_dz_mm':float((obj[-1,2]-obj[0,2])*1000),'TCP_max_positive_step_mm':float(max(0,dz.max())),'object_max_positive_step_mm':float(max(0,oz.max())),'TCP_total_positive_dz_mm':float(dz[dz>0].sum()),'object_total_positive_dz_mm':float(oz[oz>0].sum())}
  achieved=True
  if condition=='baseline':achieved=not any(q['labels']['holding_derived'])and q['labels']['categorical'][3]==(2 if root['drawer_setting']=='open'else 0)
  elif condition in ['near_open','near_closed']:achieved=not o['held']and o['external_support_contacts']>0 and o['tcp_distance_m']<.075
  elif condition=='held_after_lift':achieved=o['held']
  elif condition=='held_stationary':achieved=motion['held_all64']and max(motion['TCP_displacement_from_start_max_mm'],motion['object_displacement_from_start_max_mm'])<=1
  elif condition=='held_downward':achieved=motion['held_all64']and max(motion['TCP_net_dz_mm'],motion['object_net_dz_mm'])<=-20 and max(motion['TCP_max_positive_step_mm'],motion['object_max_positive_step_mm'])<=.02 and max(motion['TCP_total_positive_dz_mm'],motion['object_total_positive_dz_mm'])<=.10
  elif condition in ['in_open','in_closed']:achieved=q['labels']['categorical'][j]==3 and q['labels']['categorical'][3]==(2 if condition=='in_open'else 0)
  sd=folder/'samples'/condition;sd.mkdir(parents=True);lp=sd/'labels_private.json';save(lp,{'frames':[trace[i]for i in inds],'anchor_frame':anchor,'motion_audit_private':motion,'intended_condition_achieved':bool(achieved),'current_label_semantics':'unchanged current-anchor physical truth; no visibility mask'})
  records.append({'sample_id':root['root_id']+'__'+condition,'unique_window_id':root['root_id']+f'__t{anchor:04d}','root_id':root['root_id'],'environment':'D','split':'independent_acceptance','target_color':target,'initial_drawer':root['drawer_setting'],'condition':condition,'anchor_frame':anchor,'history_frame_indices':inds,'public_frames':[str(folder/'public'/f'{i:04d}.npz')for i in inds],'labels_private':str(lp),'intended_condition_achieved':bool(achieved),'new_independent_root':True,'motion':motion})
 summary={'root_id':root['root_id'],'target_color':target,'initial_drawer':root['drawer_setting'],'frames':len(trace),'controls':max(0,len(trace)-1),'attempts':attempts,'planned_conditions':planned,'available_windows':len(records),'condition_missing':missing,'condition_not_achieved':[r['condition']for r in records if not r['intended_condition_achieved']],'controller_failures':[r for r in attempts if r['status']not in ['success','executed']],'collection_error':failure,'physics_guard_passed':failure is None,'seconds':time.monotonic()-start};save(folder/'samples_manifest.json',records);save(folder/'summary.json',summary);return records,summary
def main():
 protocol=read(E/'configs/protocol.json')
 for path,h in protocol['frozen_sha256'].items():assert sha(path)==h,path
 assert sha(B/'roots_manifest.json')==protocol['new_roots_sha256'];lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);manifest=read(B/'data_manifest.json');done={r['root_id']for r in manifest['roots']}
 for root in read(B/'roots_manifest.json'):
  if root['root_id']in done:continue
  rows,s=collect(root,protocol);manifest['records']+=rows;manifest['roots'].append(s);save(B/'data_manifest.json',manifest);save(E/'results/progress.json',{'stage':'collection','collected_roots':len(manifest['roots']),'available_windows':len(manifest['records']),'last_root':root['root_id'],'training':False});print(json.dumps({'root':root['root_id'],'frames':s['frames'],'windows':len(rows),'missing':s['condition_missing'],'not_achieved':s['condition_not_achieved'],'control_failures':s['controller_failures'],'collection_error':s['collection_error']},ensure_ascii=False),flush=True)
  if s['collection_error']:raise RuntimeError('Collection infrastructure error preserved; do not silently rerun this root')
 save(E/'results/collection_complete.json',{'roots':len(manifest['roots']),'frames':sum(s['frames']for s in manifest['roots']),'windows':len(manifest['records']),'new_model_inference':False});print('INDEPENDENT_COLLECTION_COMPLETE',flush=True)
if __name__=='__main__':main()
