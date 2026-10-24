"""New-image/new-physics-label pilot from 128 fixed official initializations.

Never attaches reconstructed labels to the old official images. No learning.
"""
from pathlib import Path
import json,hashlib,datetime,sys,shutil,time,traceback,fcntl,contextlib
import numpy as np
import pybullet as pb
from scipy.spatial.transform import Rotation
from omegaconf import OmegaConf
P=Path('/home/robot/research/projects/wm_symbolic')
OLD=P/'experiments/official_AB_state_pilot_20261018'
E=P/'experiments/official_state_resimulation_20261018'
R=P/'reports/official_state_resimulation_20261018'
EV=P/'experiments/combination_generalization_20261017/evaluator_review_v1'
REP=P/'experiments/interaction_latent_repair_20261016'
sys.path.insert(0,str(EV/'source'));from native_evaluator import NativeEvaluator
sys.path.insert(0,str(REP/'source'));from labels_v2_1 import snapshot,SPEC
from wm_symbolic.env_adapter import CalvinAdapter
from wm_symbolic.physical_oracle import PhysicalOracle
COLORS=['red','blue','pink'];CHECKPOINTS=[1,4,8];STABLE_WINDOW=[4,5,6,7,8]
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text())
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def save(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=lambda x:x.tolist()if isinstance(x,np.ndarray)else x.item()if isinstance(x,np.generic)else str(x))+'\n')
def preflight():
 assert not E.exists(),'Separate pilot already exists; no automatic repeat'
 previous=read(OLD/'configs/delivery_manifest.json')
 for path,h in previous['sha256'].items():assert sha(path)==h,path
 selection=read(OLD/'configs/combined_frame_selection.json');assert selection['unique_frames']==128
 color=read(OLD/'results/color_mapping_evidence_refined.json');assert all(color[s]['native_identity_supported']for s in ['A','B'])
 roots=selection['roots'];assert len(roots)==4 and sum(len(r['frame_ids'])for r in roots)==128
 for sub in ['source','configs','data','results','logs']:(E/sub).mkdir(parents=True)
 R.mkdir(parents=True)
 shutil.copy2(__file__,E/'source/run.py')
 sourcefiles=[E/'source/run.py',P/'src/wm_symbolic/env_adapter.py',P/'src/wm_symbolic/physical_oracle.py',EV/'source/native_evaluator.py',REP/'source/labels_v2_1.py',P/'vendor/calvin/calvin_env/calvin_env/robot/robot.py',P/'vendor/calvin/calvin_env/calvin_env/robot/mixed_ik.py',P/'vendor/calvin/calvin_env/calvin_env/envs/play_table_env.py',P/'vendor/calvin/calvin_env/calvin_env/scene/play_table_scene.py',OLD/'configs/delivery_manifest.json',OLD/'configs/combined_frame_selection.json',OLD/'results/color_mapping_evidence_refined.json']
 for scene in ['A','B']:
  ev=NativeEvaluator(scene);save(E/'configs'/f'native_geometry_{scene}.json',ev.provenance);sourcefiles.append(Path(ev.provenance['scene']))
  sourcefiles.extend(Path(o['urdf'])for o in ev.provenance['objects'].values())
 inputs=[]
 for root in roots:
  for fid in root['frame_ids']:
   path=OLD/'data/frames'/root['root_id']/f'episode_{fid:07d}.npz'
   with np.load(path,allow_pickle=False)as a:assert a['robot_obs'].shape==(15,)and float(a['robot_obs'][-1])in[-1.,1.]
   inputs.append({'root_id':root['root_id'],'scene':root['scene'],'official_episode_index':root['official_episode_index'],'official_episode_inclusive':root['official_episode_inclusive'],'frame_id':fid,'sample_roles':root['frame_roles'][str(fid)],'source_path':str(path),'source_sha256':sha(path)})
 protocol={'utc':now(),'dataset_type':'official-state-initialized resimulation','scope':'new rendered RGB/proprio paired ONLY with simultaneous new complete physical state and labels; official RGB remains isolated reference','input_initializations':128,'independent_source_episodes':4,'new_download_bytes':0,'training_updates':0,'model_forward_calls':0,'fixed_controller_transition_training':False,'scene_binding':{'A':['pink','blue','red'],'B':['red','blue','pink']},'control_restore_audit':'native robot.reset sets arm targets to recorded joint angles and finger targets to observed width/2, both force200; merely assigning robot.gripper_action does not restore a command. Native apply_action is used exactly once before integration to hold recorded robot_obs TCP position/euler orientation and robot_obs[-1] grip command. Capture every issued motor target. Close uses target0 and force200, open uses target.04 and force2, according to unmodified native config. No dataset next-action motion is replayed.','initialization':'fresh native world per input; set recorded scene/robot positions once; explicitly zero unrecorded object velocities. No env.reset implicit tick. performCollisionDetection records t0; no integration before command audit.','fixed_physics_steps':8,'physics_dt_seconds':1/240,'image_checkpoints':CHECKPOINTS,'physical_record_steps':list(range(9)),'native_action_equivalence':'one fixed absolute native command, then8 physics ticks; native scene.step only at8 as in normal control period; intermediate observations at1/4 do not alter controls','candidate_selection':'ALWAYS step8. Never select a per-state best step. Retain step1/4/8 raw strict labels and new images. Candidate fact is reliable only if native value and native validity are stable at EVERY tick4..8 and force confidence is unambiguous at every tick. Otherwise unknown.','force_confidence_mask':'Holding positive: both finger families max force>=.015N and every external contact force<=.005N. Holding negative: external support force>=.015N OR at least one family max force<=.005N. In positive: frozen evaluator valid and certain contained support path; In negative: evaluator valid and value false. For geometrically contained/borderline objects also require confident Holding status. Thresholds add conservative uncertainty masks only, not predicate changes; .005/.015 are existing support uncertainty bounds.','pose_drift':'Record object translations/rotations and TCP displacement/orientation from original and initialized state at every tick; do not adjust steps or move objects to improve labels.','prohibited_after_init':['object/joint teleport','new constraint','dynamics changes','external force/torque'],'failure_policy':'Each input at most once. Persist started ledger before world reset. On technical failure preserve partial files, stop entire run and report; no silent retry or replacement.','maximum_physics_ticks':1024,'no_formal_adoption':True}
 save(E/'configs/protocol.json',protocol);save(E/'configs/inputs.json',inputs)
 save(E/'configs/frozen.json',{'utc':now(),'before_any_physics_run':True,'sha256':{str(p):sha(p)for p in dict.fromkeys(sourcefiles+[E/'configs/protocol.json',E/'configs/inputs.json'])}})
 (R/'CONTROL_AND_PLAN.md').write_text('# 控制恢复核查与固定再仿真方案\n\n源码核查确认reset只把手指电机目标设为观测宽度/2，不能由gripper_action变量赋值恢复开闭驱动。本轮每个初态只调用一次未修改的原生apply_action：保持robot_obs记录的末端位置/朝向和最后夹爪命令；逐项记录实际发给电机的目标、力及速度。原生闭合目标0、力200；张开目标0.04、力2。\n\n仅复用既有4个episode/128帧，每初态一次运行，固定8个1/240秒物理步。第1/4/8步重渲染双RGB并保存proprio；每步完整记录物体、抽屉、机器人、接触及严格物理标签。最终候选只取第8步；第4–8步值、有效性和力证据始终明确才标可靠，不一致或临界则未知。不会逐帧调步数或为正例挪动物体。技术失败保存后停止，不重试。\n\n所有新标签只关联同刻新渲染图，不关联旧官方RGB；类型为official-state-initialized resimulation，仅状态读取候选，不进入固定控制器转移训练。无新下载、模型训练或正式模型替换。\n')
 print('PREFLIGHT_FROZEN',128,'initializations, steps1/4/8, final8, stable4..8',flush=True)
 return inputs
def contact_confidence(physical,ctl,labels):
 result={}
 for j,c in enumerate(COLORS):
  uid=ctl.objects['block_'+c].uid;families=[0.,0.];external=0.
  for contact in physical['contacts']:
   if contact[1]==uid:other,link=contact[2],contact[4]
   elif contact[2]==uid:other,link=contact[1],contact[3]
   else:continue
   force=max(0.,float(contact[9]))
   if other==ctl.robot.robot_uid:
    for k,family in enumerate(ctl.finger_families):
     if link in family:families[k]=max(families[k],force)
   else:external=max(external,force)
  y=labels['y'][j]
  hconf=bool(min(families)>=.015 and external<=.005)if y else bool(external>=.015 or min(families)<=.005)
  g=labels['objects'][c];iconf=bool(labels['v'][3+j]and(g['outside']or hconf))
  if labels['y'][3+j]:iconf=iconf and bool(g['certain_contained_load_bearing_paths'])
  result[c]={'family_max_normal_force_N':families,'max_external_normal_force_N':external,'Holding_confident':hconf,'In_confident':bool(iconf)}
 return result
@contextlib.contextmanager
def prohibit_state_edits():
 names=['resetBasePositionAndOrientation','resetBaseVelocity','resetJointState','resetJointStatesMultiDof','createConstraint','changeConstraint','changeDynamics','applyExternalForce','applyExternalTorque','resetSimulation']
 originals={name:getattr(pb,name)for name in names if hasattr(pb,name)}
 def blocked(*args,**kwargs):raise RuntimeError('Prohibited physical-state intervention after initialization')
 try:
  for name in originals:setattr(pb,name,blocked)
  yield
 finally:
  for name,value in originals.items():setattr(pb,name,value)
def motor_audit(commands,robot,original):
 reset=[x for x in commands if x['phase']=='reset'and x['bodyUniqueId']==robot.robot_uid];hold=[x for x in commands if x['phase']=='hold_command'and x['bodyUniqueId']==robot.robot_uid]
 assert len(reset)==len(hold)==9,(len(reset),len(hold))
 rd={x['jointIndex']:x for x in reset};hd={x['jointIndex']:x for x in hold};grip=int(original[-1]);goal=float(robot.gripper_joint_limits[1 if grip==1 else 0]);force=float(robot.gripper_force/(100 if grip==1 else 1))
 for j,joint in enumerate(robot.arm_joint_ids):assert abs(rd[joint]['targetPosition']-float(original[7+j]))<1e-12
 for joint in robot.gripper_joint_ids:
  assert abs(rd[joint]['targetPosition']-float(original[6]/2))<1e-12
  assert hd[joint]['controlMode']==pb.POSITION_CONTROL and abs(hd[joint]['targetPosition']-goal)<1e-12 and hd[joint]['force']==force and hd[joint]['maxVelocity']==1
 assert robot.gripper_action==grip
 return {'passed':True,'original_gripper_command':grip,'observed_half_width':float(original[6]/2),'native_command_target':goal,'native_command_force':force,'max_arm_target_change_rad':max(abs(hd[j]['targetPosition']-rd[j]['targetPosition'])for j in robot.arm_joint_ids),'reset_and_command_motor_calls':commands}
def execute(env,ev,item):
 folder=E/'data'/item['root_id']/f"{item['frame_id']:07d}";folder.mkdir(parents=True,exist_ok=False)
 source=np.load(item['source_path'],allow_pickle=False);original_robot=source['robot_obs'].copy();original_scene=source['scene_obs'].copy();np.savez_compressed(folder/'original_initial_state_reference.npz',robot_obs=original_robot,scene_obs=original_scene)
 env.env.load();env.env.scene.reset(original_scene)
 calls=[];phase='reset';old=pb.setJointMotorControl2
 def capture(*args,**kw):
  q=dict(zip(['bodyUniqueId','jointIndex','controlMode'],args));q.update(kw)
  if 'bodyIndex'in q:q['bodyUniqueId']=q.pop('bodyIndex')
  calls.append({'phase':phase,**q});return old(*args,**kw)
 pb.setJointMotorControl2=capture
 try:
  env.env.robot.reset(original_robot)
  for obj in env.env.scene.movable_objects:pb.resetBaseVelocity(obj.uid,[0,0,0],[0,0,0],physicsClientId=env.env.cid)
  pb.performCollisionDetection(physicsClientId=env.env.cid);ctl=PhysicalOracle(env);robot=env.env.robot
  # No motor call is triggered by this field assignment; it only aligns t0 metadata.
  robot.gripper_action=int(original_robot[-1]);command=[original_robot[:3].copy(),original_robot[3:6].copy(),int(original_robot[-1])]
  ticks=[];initial=None
  def record(step,render=False):
   nonlocal initial
   state=env.env.get_state_obs();physical=snapshot(env,ctl);labels=ev.evaluate(physical);confidence=contact_confidence(physical,ctl,labels)
   if initial is None:initial={'state':state,'physical':physical}
   drift={}
   for c in COLORS:
    a=initial['physical']['objects']['block_'+c];b=physical['objects']['block_'+c];drift[c]={'translation_m':float(np.linalg.norm(np.array(b['position'])-a['position'])),'rotation_rad':float((Rotation.from_quat(b['quaternion'])*Rotation.from_quat(a['quaternion']).inv()).magnitude())}
   row={'physics_step':step,'sim_seconds':step/240,'state':state,'physical':physical,'labels':labels,'confidence':confidence,'object_drift_from_initial':drift,'tcp_position_from_recorded_m':float(np.linalg.norm(np.array(state['robot_obs'])[:3]-original_robot[:3])),'tcp_position_from_initialized_m':float(np.linalg.norm(np.array(state['robot_obs'])[:3]-np.array(initial['state']['robot_obs'])[:3])),'tcp_rotation_from_recorded_rad':float((Rotation.from_euler('xyz',np.array(state['robot_obs'])[3:6])*Rotation.from_euler('xyz',original_robot[3:6]).inv()).magnitude())}
   if render:
    obs=env.observe();assert np.array_equal(obs['robot_obs'],state['robot_obs'])
    path=folder/f'observation_step{step:02d}.npz';np.savez_compressed(path,**obs['rgb_obs'],**obs['depth_obs'],robot_obs=obs['robot_obs'])
    row['new_observation_path']=str(path);row['new_observation_sha256']=sha(path);row['new_RGB_vs_original_MAE_0_255']={k:float(np.abs(obs['rgb_obs'][k].astype(float)-source[k]).mean())for k in ['rgb_static','rgb_gripper']}
   save(folder/f'physics_step{step:02d}.json',row);ticks.append(row);return row
  record(0)
  with prohibit_state_edits():
   phase='hold_command';robot.apply_action(command);audit=motor_audit(calls,robot,original_robot);audit['fixed_absolute_command']=command;audit['control_calls']=1;save(folder/'control_restore_audit.json',audit)
   dt=pb.getPhysicsEngineParameters(physicsClientId=env.env.cid)['fixedTimeStep'];assert abs(dt-1/240)<1e-12
   for step in range(1,9):
    pb.stepSimulation(physicsClientId=env.env.cid)
    save(folder/'integration_ledger.json',{'completed_physics_ticks':step,'maximum_ticks':8})
    if step==8:env.env.scene.step()
    record(step,step in CHECKPOINTS)
  last=ticks[8];stable=ticks[4:9];v=[];reasons=[]
  for j in range(6):
   color=COLORS[j%3];group='Holding'if j<3 else'In';same=all(t['labels']['y'][j]==last['labels']['y'][j]for t in stable);native=all(t['labels']['v'][j]for t in stable);conf=all(t['confidence'][color][group+'_confident']for t in stable);valid=bool(same and native and conf)
   v.append(valid);reasons.append('stable_native_and_unambiguous_force_4_to_8'if valid else'changed_fact_in_4_to_8'if not same else'native_geometry_or_force_boundary'if not native else'force_threshold_confidence_ambiguous')
  candidate={'source_type':'official-state-initialized resimulation','source_root_id':item['root_id'],'source_official_frame':item['frame_id'],'new_physics_step':8,'observation_path':last['new_observation_path'],'observation_sha256':last['new_observation_sha256'],'physics_path':str(folder/'physics_step08.json'),'fact_order':[f'Holding({c})'for c in COLORS]+[f'In({c},drawer)'for c in COLORS],'y':last['labels']['y'][:6],'v':v,'validity_reason':reasons,'stable_window':STABLE_WINDOW,'old_official_RGB_not_labeled':True,'eligible_fixed_controller_transition':False,'formal_training_enabled':False}
  save(folder/'candidate_state_label.json',candidate)
  return {**item,'folder':str(folder),'status':'completed','physics_steps':8,'candidate':candidate,'motor_audit':{k:audit[k]for k in ['passed','original_gripper_command','observed_half_width','native_command_target','native_command_force','max_arm_target_change_rad','control_calls']},'steps':[{k:t[k]for k in ['physics_step','sim_seconds','labels','confidence','object_drift_from_initial','tcp_position_from_recorded_m','tcp_position_from_initialized_m','tcp_rotation_from_recorded_rad']+(['new_RGB_vs_original_MAE_0_255','new_observation_path']if t['physics_step']in CHECKPOINTS else[])}for t in ticks]}
 finally:pb.setJointMotorControl2=old
def main():
 inputs=preflight();lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);attempts=[];results=[];start=time.monotonic();stop=None;env=None
 for scene in ['A','B']:
  try:
   env=CalvinAdapter(P,scene,170818,segmentation=False);ev=NativeEvaluator(scene);OmegaConf.save(env.config,E/'configs'/f'resolved_native_{scene}.yaml')
   for item in [r for r in inputs if r['scene']==scene]:
    attempts.append({**item,'utc':now(),'status':'started'});save(E/'results/attempts.json',attempts)
    try:
     result=execute(env,ev,item);results.append(result);attempts[-1]['status']='completed';save(E/'results/attempts.json',attempts);save(E/'results/results.json',results)
    except Exception as error:
     attempts[-1].update(status='technical_failure',error=str(error));stop={'item':item,'error':str(error),'traceback':traceback.format_exc()};save(E/'results/attempts.json',attempts);save(E/'results/technical_failure.json',stop);break
    if len(results)%16==0:print('RESIMULATED',len(results),'/128',item['root_id'],flush=True)
  except Exception as error:
   if stop is None:stop={'scene':scene,'error':str(error),'traceback':traceback.format_exc()};save(E/'results/technical_failure.json',stop)
  finally:
   if env is not None:env.close();env=None
  if stop is not None:break
 total_ticks=sum(read(p)['completed_physics_ticks']for p in(E/'data').rglob('integration_ledger.json'))
 save(E/'results/run_budget.json',{'utc':now(),'attempted_initializations':len(attempts),'completed_initializations':len(results),'known_completed_physics_ticks':total_ticks,'maximum_initializations':128,'max_ticks_per_initialization':8,'max_total_ticks':1024,'elapsed_seconds':time.monotonic()-start,'retry_count':0,'stop':stop,'new_download_bytes':0,'training_updates':0,'new_images_only_for_new_labels':True})
 if stop is not None:raise RuntimeError('Stopped on technical failure; no retry')
 assert len(results)==128;print('RESIMULATION_COMPLETE',flush=True)
if __name__=='__main__':main()
