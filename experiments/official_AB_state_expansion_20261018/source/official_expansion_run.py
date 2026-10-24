"""Once-only 96-tick official-state-initialized A/B resimulation. No model."""
from pathlib import Path
import json,gzip,shutil,datetime,time,traceback,fcntl
import numpy as np
import pybullet as pb
from scipy.spatial.transform import Rotation
from omegaconf import OmegaConf
import official_state_resimulation as S
from official_expansion_fetch import E,R,read,write
from expansion_stability import evidence,classify
P=S.P;CHECK=[32,64,96]
def serialize(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 raise TypeError(type(x))
def preflight():
 assert not(E/'results/attempts.json').exists(),'Never replay attempted frozen inputs'
 assert read(E/'results/retrieval_complete.json')['total_runs']==256
 for p,h in read(E/'configs/frozen_selection.json')['sha256'].items():assert S.sha(p)==h,p
 inputs=read(E/'configs/realized_inputs.json');assert len(inputs)==256
 for r in inputs:assert r['scene']in['A','B']and S.sha(r['source_path'])==r['source_sha256']
 for p in[Path(__file__),Path(S.__file__),Path(__file__).with_name('expansion_stability.py')]:shutil.copy2(p,E/'source'/p.name)
 refs=[E/'source'/n for n in['official_expansion_run.py','official_state_resimulation.py','expansion_stability.py']]+[E/'configs/realized_inputs.json',E/'configs/frozen_selection.json',P/'src/wm_symbolic/env_adapter.py',P/'src/wm_symbolic/physical_oracle.py',S.EV/'source/native_evaluator.py',S.REP/'source/labels_v2_1.py',P/'vendor/calvin/calvin_env/calvin_env/robot/robot.py',P/'vendor/calvin/calvin_env/calvin_env/envs/play_table_env.py']
 for scene in ['A','B']:
  ev=S.NativeEvaluator(scene);write(E/'configs'/f'native_geometry_{scene}.json',ev.provenance);refs.append(Path(ev.provenance['scene']));refs.extend(Path(o['urdf'])for o in ev.provenance['objects'].values())
 write(E/'configs/frozen_execution.json',{'utc':S.now(),'before_any_selected_initialization':True,'sha256':{str(p):S.sha(p)for p in dict.fromkeys(refs)}})
 print('EXECUTION_FROZEN',len(inputs),'runs x96',flush=True);return inputs
def execute(adapter,ev,item):
 folder=E/'data/resim'/item['root_id']/f"{item['frame_id']:07d}";folder.mkdir(parents=True,exist_ok=False)
 with np.load(item['source_path'],allow_pickle=False)as source:
  original_robot=source['robot_obs'].copy();original_scene=source['scene_obs'].copy();original_rgb={k:source[k].copy()for k in['rgb_static','rgb_gripper']}
 np.savez_compressed(folder/'original_state_reference.npz',robot_obs=original_robot,scene_obs=original_scene)
 adapter.env.load();adapter.env.scene.reset(original_scene);calls=[];phase='reset';period=-1;original_motor=pb.setJointMotorControl2
 def capture(*args,**kwargs):
  q=dict(zip(['bodyUniqueId','jointIndex','controlMode'],args));q.update(kwargs)
  if 'bodyIndex'in q:q['bodyUniqueId']=q.pop('bodyIndex')
  calls.append({'phase':phase,'control_period':period,**q});return original_motor(*args,**kwargs)
 pb.setJointMotorControl2=capture;ticks=[];initial=None;summary=[];period_audits=[]
 try:
  adapter.env.robot.reset(original_robot)
  for obj in adapter.env.scene.movable_objects:pb.resetBaseVelocity(obj.uid,[0,0,0],[0,0,0],physicsClientId=adapter.env.cid)
  pb.performCollisionDetection(physicsClientId=adapter.env.cid);ctl=S.PhysicalOracle(adapter);robot=adapter.env.robot;robot.gripper_action=int(original_robot[-1]);command=[original_robot[:3].copy(),original_robot[3:6].copy(),int(original_robot[-1])]
  history=folder/'physics_history.jsonl.gz'
  with gzip.open(history,'wt',encoding='utf-8',compresslevel=4)as stream:
   def record(step):
    nonlocal initial
    state=adapter.env.get_state_obs();physical=S.snapshot(adapter,ctl);physical.update(robot_uid=robot.robot_uid,finger_families=[sorted(x)for x in ctl.finger_families]);labels=ev.evaluate(physical);confidence=evidence(physical,labels)
    if initial is None:initial={'state':state,'physical':physical}
    drift={}
    for c in S.COLORS:
     a=initial['physical']['objects']['block_'+c];b=physical['objects']['block_'+c];drift[c]={'translation_m':float(np.linalg.norm(np.asarray(b['position'])-a['position'])),'rotation_rad':float((Rotation.from_quat(b['quaternion'])*Rotation.from_quat(a['quaternion']).inv()).magnitude())}
    row={'physics_step':step,'sim_seconds':step/240,'state':state,'physical':physical,'labels':labels,'confidence':confidence,'object_drift_from_initial':drift,'tcp_position_from_recorded_m':float(np.linalg.norm(np.asarray(state['robot_obs'])[:3]-original_robot[:3])),'tcp_position_from_initialized_m':float(np.linalg.norm(np.asarray(state['robot_obs'])[:3]-np.asarray(initial['state']['robot_obs'])[:3])),'tcp_rotation_from_recorded_rad':float((Rotation.from_euler('xyz',np.asarray(state['robot_obs'])[3:6])*Rotation.from_euler('xyz',original_robot[3:6]).inv()).magnitude())}
    if step in CHECK:
     obs=adapter.observe();assert np.array_equal(obs['robot_obs'],state['robot_obs']);path=folder/f'observation_step{step:02d}.npz';np.savez_compressed(path,**obs['rgb_obs'],**obs['depth_obs'],robot_obs=obs['robot_obs']);row.update(new_observation_path=str(path),new_observation_sha256=S.sha(path),new_RGB_vs_original_MAE_0_255={k:float(np.abs(obs['rgb_obs'][k].astype(float)-original_rgb[k]).mean())for k in original_rgb})
    stream.write(json.dumps(row,default=serialize,separators=(',',':'))+'\n');stream.flush();ticks.append(row)
    if step in [0,1,8,32,64,96]:summary.append({k:row[k]for k in['physics_step','sim_seconds','object_drift_from_initial','tcp_position_from_recorded_m','tcp_position_from_initialized_m']+(['new_RGB_vs_original_MAE_0_255','new_observation_path']if step in CHECK else[])}|{'y':labels['y'],'v':labels['v']})
   record(0)
   with S.prohibit_state_edits():
    dt=pb.getPhysicsEngineParameters(physicsClientId=adapter.env.cid)['fixedTimeStep'];assert abs(dt-1/240)<1e-12
    for step in range(1,97):
     if (step-1)%8==0:
      period=(step-1)//8;phase='hold_command';robot.apply_action(command)
      subset=[c for c in calls if c['phase']=='reset'or c['control_period']==period];audit=S.motor_audit(subset,robot,original_robot);period_audits.append({k:v for k,v in audit.items()if k!='reset_and_command_motor_calls'}|{'control_period':period,'before_physics_step':step})
      write(folder/'control_audit.json',{'fixed_absolute_command':command,'motor_calls':calls,'period_audits':period_audits,'control_calls':len(period_audits),'all_original_targets_unchanged':True})
     pb.stepSimulation(physicsClientId=adapter.env.cid);write(folder/'integration_ledger.json',{'completed_physics_ticks':step,'maximum_ticks':96})
     if step%8==0:adapter.env.scene.step()
     record(step)
  candidate=classify(ticks)|{'source_type':'official-state-initialized resimulation','source_root_id':item['root_id'],'source_frame_id':item['frame_id'],'split':item['split'],'source_kind':item['source_kind'],'observation_path':ticks[96]['new_observation_path'],'observation_sha256':ticks[96]['new_observation_sha256'],'physics_history_path':str(history),'physics_history_sha256':S.sha(history),'old_official_RGB_not_labeled':True,'eligible_fixed_controller_transition':False,'formal_training_enabled':False}
  write(folder/'candidate_state_label.json',candidate)
  return {**item,'folder':str(folder),'status':'completed','physics_steps':96,'candidate':candidate,'checkpoints':summary,'max_object_drift_all_ticks_m':max(d['translation_m']for t in ticks for d in t['object_drift_from_initial'].values()),'max_object_rotation_all_ticks_rad':max(d['rotation_rad']for t in ticks for d in t['object_drift_from_initial'].values()),'max_TCP_drift_all_ticks_m':max(t['tcp_position_from_initialized_m']for t in ticks),'motor_control_calls':12,'motor_calls':len(calls),'motor_audit_passed':all(a['passed']for a in period_audits)}
 finally:pb.setJointMotorControl2=original_motor
def main():
 inputs=preflight();lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);results=[];attempts=[];start=time.monotonic();stop=None;adapter=None
 for scene in['A','B']:
  try:
   adapter=S.CalvinAdapter(P,scene,170819,segmentation=False);ev=S.NativeEvaluator(scene);OmegaConf.save(adapter.config,E/'configs'/f'resolved_native_{scene}.yaml')
   for item in[r for r in inputs if r['scene']==scene]:
    attempts.append({k:item[k]for k in['root_id','frame_id','split','source_kind']}|{'utc':S.now(),'status':'started'});write(E/'results/attempts.json',attempts)
    try:
     result=execute(adapter,ev,item);results.append(result);attempts[-1]['status']='completed';write(E/'results/attempts.json',attempts);write(E/'results/results.json',results)
    except Exception as error:
     attempts[-1].update(status='technical_failure',error=str(error));stop={'item':item,'error':str(error),'traceback':traceback.format_exc()};write(E/'results/attempts.json',attempts);write(E/'results/technical_failure.json',stop);break
    if len(results)%16==0:print('EXPANSION_RESIMULATED',len(results),'/256',flush=True)
  except Exception as error:
   if stop is None:stop={'scene':scene,'error':str(error),'traceback':traceback.format_exc()};write(E/'results/technical_failure.json',stop)
  finally:
   if adapter is not None:adapter.close();adapter=None
  if stop is not None:break
 ticks=sum(read(p)['completed_physics_ticks']for p in(E/'data/resim').rglob('integration_ledger.json'))
 write(E/'results/run_budget.json',{'utc':S.now(),'attempted_initializations':len(attempts),'completed_initializations':len(results),'known_completed_physics_ticks':ticks,'maximum_initializations':256,'max_ticks_per_initialization':96,'max_total_ticks':24576,'elapsed_seconds':time.monotonic()-start,'retry_count':0,'stop':stop,'training_updates':0,'model_forward_calls':0,'new_images_only_for_new_labels':True,'cumulative_download_body_bytes':read(E/'results/network_ledger.json')['body_bytes_read']})
 if stop:raise RuntimeError('Technical failure; preserved and stopped without retry')
 assert len(results)==256 and ticks==24576;print('EXPANSION_RESIMULATION_COMPLETE',flush=True)
if __name__=='__main__':main()
