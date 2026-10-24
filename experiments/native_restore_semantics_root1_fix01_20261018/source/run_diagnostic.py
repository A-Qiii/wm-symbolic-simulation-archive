from common import *
import sys,copy,enum,fcntl,time,traceback
from collections import deque
sys.path.append(str(OLD/'source'))
from collect import Runtime,Adapter,compare,encoded,ForkInvalid
import pybullet as pb
import hydra
from omegaconf import OmegaConf
from wm_symbolic.physical_oracle_v2 import PhysicalOracleV2
from wm_symbolic.physical_oracle import ExecutionFailure
import labels_physical as truth
sys.path.append(str(P/'scripts'))
from evaluate_physical_oracle_v2 import physics_guard

def split_state(x):
 """Preserve all fields; classify prior solver outputs separately, never discard."""
 x=json.loads(encoded(x));reported=[]
 for body in x['bodies']:
  for j,st in enumerate(body['joint_states']):
   st=list(st);reported.append({'body':body['uid'],'joint':j,'applied_motor_effort':st[3]});st[3]='prior_step_reported_motor_effort';body['joint_states'][j]=st
 for c in x['constraints']:reported.append({'constraint':c['id'],'reported_applied_forces':c['applied_forces']});c['applied_forces']='prior_step_reported_constraint_force'
 return x,reported

class Diagnostic(Runtime):
 def __init__(self):
  self.rid='fork12_A_00';self.spec=read(OLD/'configs/protocol.json')['roots'][0];self.d=E/'private';self.env=None;self.ctl=None;self.mode='initialize';self.ticks=0;self.actual_ticks=0;self.total_controls=0;self.condition_controls=0;self.hist=deque(maxlen=12);self.motors={};self.caps=[];self.snapshots=[];self.tracefile=None;self.ref_monitor=None;self.safety_stats={};self.branch=None;self.local_step=None;self.prefix=[];self.records=[];self.fidelity=[];self.run_name=None;self.tick_within_control=0;self.first_tick_capture=False
  self.original={n:getattr(pb,n)for n in ['getCameraImage','stepSimulation','setJointMotorControl2']};pb.stepSimulation=self.step_sim;pb.getCameraImage=self.image;pb.setJointMotorControl2=self.motor
  cfg=OmegaConf.load(OLDROOT/'resolved_config.yaml');self.env=hydra.utils.instantiate(cfg.env,use_scene_info=True);pb.setRealTimeSimulation(0,physicsClientId=self.env.cid);assert self.env.action_repeat==8 and not self.env.robot.use_ik_fast and self.env.cid==0
  saved=self.env.get_obs;self.env.get_obs=self.env.get_state_obs
  try:self.env.reset()
  finally:self.env.get_obs=saved
  self.adapter=Adapter(self);self.ctl=PhysicalOracleV2(self.adapter,self.record,max_controls=900);self.mapping=read(OLDROOT/'body_link_mapping.json');assert self.env.robot.robot_uid==self.mapping['robot_uid'];assert list(self.env.robot.gripper_joint_ids)==self.mapping['finger_joint_ids'];self.mode='checkpoint'
  archived=read(OLDROOT/'aligned_action_start_checkpoint_private.json');pb.restoreState(fileName=str(OLDROOT/'aligned_action_start.bullet'),physicsClientId=self.env.cid);self.restore_python(self.rehydrate(archived['python']));self.mode='execute'
  actual=self.physical();a,ad=split_state(archived['physical']);b,bd=split_state(actual);self.archive_bridge={'core':compare(a,b),'reported_outputs':compare(ad,bd),'full':compare(archived['physical'],actual),'python':compare(archived['python'],self.python_state()),'actual_initialization_ticks':self.actual_ticks};save(E/'results/archive_bridge.json',self.archive_bridge);save(self.d/'archive_restored_physical.json',actual)
  self.capture(E/'private/archive_bridge_images')
  original_images=OLDROOT/'branches/fork12_A_00_aligned_lift/s00';images=[]
  for cam in ['static','gripper']:
   with np.load(original_images/f'native_{cam}.npz',allow_pickle=False)as x,np.load(E/'private/archive_bridge_images'/f'native_{cam}.npz',allow_pickle=False)as y:
    for key in ['rgb','depth_buffer','mask']:images.append({'camera':cam,'field':key,'equal':np.array_equal(x[key],y[key]),'different_values':int(np.sum(x[key]!=y[key]))})
  self.archive_bridge['images']=images;save(E/'results/archive_bridge.json',self.archive_bridge)
  if not(self.archive_bridge['core']['equal']and self.archive_bridge['python']['equal']and all(i['equal']for i in images)):raise ForkInvalid('archive_materialization_not_exact_in_evolving_state_or_observations')
  self.cp=self.checkpoint('diagnostic_native_S');self.initialization_ticks=self.actual_ticks;self.mode='execute'
 def rehydrate(self,x):
  x=copy.deepcopy(x)
  for group,items in x['scene'].items():
   for obj,vals in zip(getattr(self.env.scene,group),items):
    for k,v in vals.items():
     current=getattr(obj,k)
     if isinstance(current,enum.Enum):vals[k]=type(current)[v.split('.')[-1]]
  for group,obj in [('robot',self.env.robot),('mixed_ik',self.env.robot.mixed_ik)]:
   for k,v in x[group].items():
    if isinstance(getattr(obj,k),np.ndarray):x[group][k]=np.array(v,dtype=getattr(obj,k).dtype)
  if x['controller']['monitor']:
   x['controller']['monitor']['scene']=np.asarray(x['controller']['monitor']['scene'])
   x['controller']['monitor']['objects']={k:np.asarray(v)for k,v in x['controller']['monitor']['objects'].items()}
  return x
 def step_sim(self,*a,**kw):
  assert self.mode in ['initialize','execute'];self.ticks+=1;self.actual_ticks+=1;out=self.original['stepSimulation'](*a,**kw)
  if self.run_name is not None:
   self.tick_within_control+=1
   if self.local_step==1 and self.tick_within_control<=8:
    save(E/'private/runs'/self.run_name/'first_control_ticks'/f'tick{self.tick_within_control:02d}.json',{'physical':self.physical(),'cutoffs':self.cutoff_labels(),'tick':self.ticks,'relative_tick':self.tick_within_control})
    if self.tick_within_control==1:self.capture(E/'private/runs'/self.run_name/'first_physics_tick_images')
  return out
 def capture(self,sd):
  sd=Path(sd);sd.mkdir(parents=True,exist_ok=False);self.caps=[];before=self.physical();oldmode=self.mode;self.mode='primary';self.env.get_obs();self.mode='baseline'
  for cam in self.env.cameras:cam.render()
  after=self.physical();assert compare(before,after)['equal']and len(self.caps)==4 and len({c['meta']['tick']for c in self.caps})==1
  for cam,a,b in zip(self.env.cameras,self.caps[:2],self.caps[2:]):
   assert np.array_equal(a['rgb'],b['rgb'])and np.array_equal(a['depth_buffer'],b['depth_buffer']);save_raw_npz(sd/f'native_{cam.name}.npz',rgb=a['rgb'],depth_buffer=a['depth_buffer'],mask=a['mask']);save(sd/f'camera_{cam.name}.json',a['meta'])
  save(sd/'integrity.json',{'same_tick':True,'no_physics_during_render':True,'RGB_depth_equal_without_mask_flag':True,'tick':self.ticks});self.mode=oldmode
 def record(self,obs,action):
  phy=truth.snapshot(self.adapter,self.ctl);cuts=self.cutoff_labels();self.hist.append({'control':self.adapter.controls,'tick':self.ticks,'physical':phy,'holding_at_cutoffs':cuts})
  robot=self.env.robot.robot_uid;target=self.mapping['object_uids']['red'];cs=[c for c in phy['contacts']if c[9]>.01 and robot in[c[1],c[2]]];bad=[c for c in cs if(c[2]if c[1]==robot else c[1])!=target];penetration=max([max(0,-c[8])for c in cs],default=0.);entry={'run':self.run_name,'control':self.local_step,'action':action,'physical':self.physical(),'truth':phy,'cutoffs':cuts,'history':list(self.hist),'python_state':self.python_state(),'max_robot_penetration_m':penetration,'non_target_robot_contacts':bad};save(E/'private/runs'/self.run_name/f'c{self.local_step:02d}.json',entry)
  if penetration>=.01 or bad:raise ExecutionFailure('safety_contact_or_penetration')
 def run(self,spec):
  self.run_name=None;self.local_step=None;self.mode='checkpoint';prior=self.physical();pb.restoreState(stateId=self.cp['id'],physicsClientId=self.env.cid);native=self.physical();self.restore_python(self.cp['python']);post=self.physical();self.mode='execute';a,ad=split_state(self.cp['physical']);b,bd=split_state(post);result={'name':spec['name'],'primitive':spec['primitive'],'previous_history':spec['previous_history'],'status':'started','executed_controls':0,'start_core':compare(a,b),'start_reported':compare(ad,bd),'start_full':compare(self.cp['physical'],post),'python':compare(self.cp['python'],self.python_state()),'motor_reissue_physical_effect':compare(native,post)};run=E/'private/runs'/spec['name'];run.mkdir();save(run/'restore_stages.json',{'before_restore':prior,'after_native_before_motor_reissue':native,'after_motor_and_python_restore':post,'comparisons':result});self.capture(run/'c00_images');save(run/'c00.json',{'control':0,'physical':post,'cutoffs':self.cutoff_labels(),'history':list(self.hist),'python_state':self.python_state()})
  if not(result['start_core']['equal']and result['python']['equal']and result['motor_reissue_physical_effect']['equal']):raise ForkInvalid('native_checkpoint_core_or_python_restore_failed:'+spec['name'])
  commands=np.array(read(E/'configs/action_programs.json')[spec['primitive']]);self.run_name=spec['name'];self.condition_controls=0;start=self.total_controls
  try:
   with physics_guard(self.adapter,self.ctl):
    for k,command in enumerate(commands,1):
     self.local_step=k;self.tick_within_control=0;self.ctl.phase='restore_semantics_fixed_prefix';self.ctl.step(command[:3],command[3:7],int(command[7]));self.capture(run/f'c{k:02d}_images')
   result['status']='completed'
  finally:
   result['executed_controls']=self.total_controls-start;save(run/'result.json',result);self.run_name=None
  return result

def main():
 assert not(E/'results/started.json').exists()
 for p,h in read(E/'configs/frozen.json')['sha256'].items():assert sha(p)==h,p
 lock=(P/'runs/gpu.lock').open('r+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);save(E/'results/started.json',{'utc':now()});rt=None;results=[];stop=None;t=time.perf_counter()
 try:
  rt=Diagnostic.__new__(Diagnostic);rt.__init__()
  for spec in read(E/'configs/protocol.json')['runs']:
   result=rt.run(spec);results.append(result);save(E/'results/runs.json',results);print('PREFIX',result['name'],result['status'],result['executed_controls'],flush=True)
 except Exception as exc:
  stop={'exception':repr(exc),'traceback':traceback.format_exc(),'utc':now()};save(E/'results/STOPPED.json',stop)
 finally:
  summary={'utc':now(),'runs_completed':len(results),'execution_controls':rt.total_controls if rt else 0,'actual_ticks':rt.actual_ticks if rt else 0,'stop':stop,'seconds':time.perf_counter()-t,'fits':0,'new_layouts':0,'action_effect_interpretation':False}
  save(E/'results/execution_summary.json',summary)
  if rt:rt.close()
  lock.close()
 print('DIAGNOSTIC_END',summary,flush=True)
if __name__=='__main__':main()
