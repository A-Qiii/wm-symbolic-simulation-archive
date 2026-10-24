from common import *
from safety import physical_guard,opening_sweep
import sys,time,fcntl,traceback,uuid,copy,platform
from collections import deque
import pybullet as pb
import hydra
from hydra import initialize_config_dir,compose
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation
from wm_symbolic.physical_oracle_v2 import PhysicalOracleV2
from wm_symbolic.physical_oracle import ExecutionFailure

HISTORICAL_RGB=set();SEEN_RGB={}
class MotorAdapter:
 def __init__(self,rt):self.rt=rt;self.env=rt.env;self.controls=0;self.last_grip=1
 def observe(self):return {'robot_obs':self.env.get_state_obs()['robot_obs']}
 def step(self,action):
  if self.controls>=900 or(self.rt.stage!='action'and self.controls>=828):raise ExecutionFailure('fixed_control_budget_exhausted')
  cmd=np.r_[np.asarray(action[0]),np.asarray(action[1]),action[2]]
  if self.rt.stage=='action':assert np.array_equal(cmd,self.rt.program[len(self.rt.actual_actions)]),'preset_action_mismatch'
  self.env.robot.apply_action(action)
  for _ in range(8):pb.stepSimulation(physicsClientId=self.env.cid)
  self.env.scene.step();self.controls+=1;self.last_grip=int(action[2])
  if self.rt.stage=='action':self.rt.actual_actions.append(cmd.tolist())
  return self.observe()

class Runtime:
 def __init__(self,spec):
  self.spec=spec;self.rid=spec['root_id'];self.d=E/'data/private'/self.rid;self.d.mkdir(exist_ok=False)
  self.env=None;self.ctl=None;self.adapter=None;self.mode='initialize';self.stage='initialization';self.ticks=0;self.initial_ticks=None;self.camera_calls=0;self.hist=deque(maxlen=12);self.captures=[];self.actual_actions=[];self.program=None;self.pre_input=None;self.caps=[];self.ref=None;self.guard_audit={'forbidden_attempts':[],'shadow_only_reset_calls':0,'native_robot_motor_calls':0};self.safety={'max_robot_penetration_m':0.,'non_target_robot_contact_samples':0,'max_device_drift':0.,'max_non_target_displacement_m':0.};self.orig={n:getattr(pb,n)for n in ['stepSimulation','getCameraImage']};self.trace=(self.d/'controls_private.jsonl').open('w');pb.stepSimulation=self.physics_step;pb.getCameraImage=self.image
 def initialize(self):
  s=self.spec
  with initialize_config_dir(config_dir=str(P/'vendor/calvin/calvin_env/conf'),job_name='forward54'):
   cfg=compose(config_name='config_data_collection',overrides=[f'scene=calvin_scene_{s["scene"]}','cameras=static_and_gripper','use_vr=false',f'env.seed={s["seed"]}','env.use_egl=true'])
  cfg.scene.objects.fixed_objects.table.joints.base__drawer.initial_state=.005
  for c,b in s['blocks'].items():cfg.scene.objects.movable_objects['block_'+c].initial_pos=b['initial_pos'];cfg.scene.objects.movable_objects['block_'+c].initial_orn=b['initial_orn']
  OmegaConf.save(cfg,self.d/'resolved_config.yaml');self.env=hydra.utils.instantiate(cfg.env,use_scene_info=True);pb.setRealTimeSimulation(0,physicsClientId=self.env.cid)
  assert self.env.action_repeat==8 and not self.env.robot.use_ik_fast
  getobs=self.env.get_obs;self.env.get_obs=self.env.get_state_obs
  try:self.env.reset()
  finally:self.env.get_obs=getobs
  self.initial_ticks=self.ticks;self.adapter=MotorAdapter(self);self.ctl=PhysicalOracleV2(self.adapter,self.record,max_controls=900)
  self.mapping={'robot_uid':self.env.robot.robot_uid,'object_uids':{c:self.ctl.objects['block_'+c].uid for c in COLORS},'finger_families':[sorted(f)for f in self.ctl.finger_families],'table_uid':self.ctl.table.uid}
  assert len(self.mapping['finger_families'])==2 and all(len(f)==2 for f in self.mapping['finger_families'])
  save(self.d/'mapping.json',self.mapping);save(self.d/'physics_parameters.json',pb.getPhysicsEngineParameters(physicsClientId=self.env.cid))
  self.mode='execute';self.ref={'scene':self.env.scene.get_obs().copy(),'objects':{n:self.ctl.xyz(n).copy()for n in self.ctl.objects},'allowed':{'block_'+s['target_color']}};self.ctl.monitor=copy.deepcopy(self.ref)
 def physics_step(self,*a,**kw):
  assert self.mode in ['initialize','execute'],'physics_during_capture_or_geometry_query'
  if self.env is not None:assert kw.get('physicsClientId',0)==self.env.cid,'query_world_physics_forbidden'
  self.ticks+=1;return self.orig['stepSimulation'](*a,**kw)
 def image(self,*a,**kw):
  assert self.mode in ['primary','baseline'],'unexpected_intermediate_render'
  kw['flags']=pb.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX if self.mode=='primary'else 0
  im=self.orig['getCameraImage'](*a,**kw);w,h=map(int,im[:2]);self.camera_calls+=1
  self.caps.append({'rgb':np.asarray(im[2],np.uint8).reshape(h,w,4)[:,:,:3].copy(),'depth':np.asarray(im[3],np.float32).reshape(h,w).copy(),'mask':np.asarray(im[4],np.int32).reshape(h,w).copy(),'meta':{'tick':self.ticks,'width':w,'height':h,'viewMatrix':kw['viewMatrix'],'projectionMatrix':kw['projectionMatrix'],'flags':kw['flags']}});return im
 def physical(self):
  cid=self.env.cid;out=[]
  for k in range(pb.getNumBodies(physicsClientId=cid)):
   u=pb.getBodyUniqueId(k,physicsClientId=cid);n=pb.getNumJoints(u,physicsClientId=cid)
   out.append({'uid':u,'pose':pb.getBasePositionAndOrientation(u,physicsClientId=cid),'velocity':pb.getBaseVelocity(u,physicsClientId=cid),'joints':pb.getJointStates(u,list(range(n)),physicsClientId=cid)if n else [],'links':[pb.getLinkState(u,j,computeLinkVelocity=1,computeForwardKinematics=1,physicsClientId=cid)for j in range(n)]})
  return {'bodies':out,'contacts':sorted([list(c)for c in pb.getContactPoints(physicsClientId=cid)],key=encoded)}
 def record(self,obs,action):
  cs=[list(c)for c in pb.getContactPoints(physicsClientId=self.env.cid)];features=contact_features(cs,self.mapping);state=self.env.scene.get_obs();objects={n:self.ctl.xyz(n).tolist()for n in self.ctl.objects}
  h={'control':self.adapter.controls,'tick':self.ticks,'contacts':cs,'features':features};self.hist.append(h)
  self.trace.write(encoded({**h,'stage_private':self.stage,'action_control':len(self.actual_actions)if self.stage=='action'else None,'actual_command':action,'robot_obs_private':obs['robot_obs'],'scene_obs_private':state,'object_positions_private':objects})+'\n');self.trace.flush()
  contacts=[c for c in cs if c[9]>.01 and self.mapping['robot_uid']in[c[1],c[2]]];target=self.mapping['object_uids'][self.spec['target_color']]
  bad=[c for c in contacts if(c[2]if c[1]==self.mapping['robot_uid']else c[1])!=target];penetration=max([max(0.,-c[8])for c in contacts],default=0.)
  drift=float(np.max(np.abs(state[[0,2,3,4,5]]-self.ref['scene'][[0,2,3,4,5]])));disp=max((float(np.linalg.norm(np.asarray(objects[n])-v))for n,v in self.ref['objects'].items()if n not in self.ref['allowed']),default=0.)
  self.safety['max_robot_penetration_m']=max(self.safety['max_robot_penetration_m'],penetration);self.safety['non_target_robot_contact_samples']+=len(bad);self.safety['max_device_drift']=max(self.safety['max_device_drift'],drift);self.safety['max_non_target_displacement_m']=max(self.safety['max_non_target_displacement_m'],disp)
  if penetration>=.01:raise ExecutionFailure('robot_penetration_at_least_10mm')
  if bad:raise ExecutionFailure('non_target_robot_contact_over_0.01N')
  if drift>.003:raise ExecutionFailure('unrequested_device_drift_over_3mm')
  if disp>.01:raise ExecutionFailure('unrequested_object_displacement_over_10mm')
 def capture(self,anchor):
  assert len(self.captures)<6
  if self.captures and self.captures[-1]['control']==self.adapter.controls:return self.captures[-1]
  before=self.physical();obs=self.adapter.observe()['robot_obs'].copy();hist=list(self.hist)
  if not hist:hist=[{'control':self.adapter.controls,'tick':self.ticks,'contacts':before['contacts'],'features':contact_features(before['contacts'],self.mapping)}]
  labels=labels_from_history(hist);self.caps=[];self.mode='primary';self.env.get_obs();self.mode='baseline'
  for cam in self.env.cameras:cam.render()
  self.mode='frozen';after=self.physical();assert encoded(before)==encoded(after),'render_changed_physics'
  assert len(self.caps)==4 and len({c['meta']['tick']for c in self.caps})==1
  for a,b in zip(self.caps[:2],self.caps[2:]):assert np.array_equal(a['rgb'],b['rgb'])and np.array_equal(a['depth'],b['depth'])
  sd=self.d/f's{len(self.captures):02d}';sd.mkdir();views={};hashes=[]
  for cam,c in zip(['static','gripper'],self.caps[:2]):
   npz(sd/f'native_{cam}.npz',rgb=c['rgb'],depth_buffer=c['depth'],object_link_mask=c['mask']);save(sd/f'camera_{cam}.json',c['meta']);h=hashlib.sha256(c['rgb'].tobytes()).hexdigest();hashes.append(h)
   # Save the attempted root's evidence before reporting any lineage blocker.
   seg=c['mask'];body=seg&((1<<24)-1);link=(seg>>24)-1;fp=[int(((seg>=0)&(body==self.mapping['robot_uid'])&np.isin(link,f)).sum())for f in self.mapping['finger_families']]
   views[cam]={color:{'object_pixels':int(((seg>=0)&(body==uid)).sum()),'finger_pixels':fp,'RGB_sha256':h}for color,uid in self.mapping['object_uids'].items()}
  for color in COLORS:
   visible=any(v[color]['object_pixels']>=4 and all(n>=4 for n in v[color]['finger_pixels'])for v in views.values());l=labels[color];l['visible']=bool(visible);l['eligible']=bool(visible and l['holding_stable']);l['unknown_reasons']=([]if l['holding_stable']else['force_or_temporal_ambiguity'])+([]if visible else['visibility_below_4pixels']);l['supported_hard_negative']=bool(l['eligible']and not l['Holding']and l['SupportTable']and l['support_stable']and l['bilateral']and l['bilateral_stable'])
  cap={'anchor':anchor,'control':self.adapter.controls,'action_control':len(self.actual_actions),'tick':self.ticks,'path':str(sd),'labels':labels,'views':views,'same_tick':True,'render_physics_unchanged':True,'RGB_depth_baseline_equal':True,'robot_obs':obs.tolist()}
  save(sd/'physical_before.json',before);save(sd/'physical_after.json',after);save(sd/'annotations.json',{'labels':labels,'last12_controls':hist,'views':views});save(sd/'capture.json',cap);self.captures.append(cap);save(self.d/'captures.json',self.captures)
  for h in hashes:
   assert h not in HISTORICAL_RGB,'historical_RGB_duplicate'
   assert h not in SEEN_RGB or SEEN_RGB[h]==self.rid,'cross_root_RGB_duplicate'
   SEEN_RGB[h]=self.rid
  self.mode='execute';return cap
 def checked(self,waypoints,name,filename):
  self.mode='geometry';q=self.ctl.shadow.check(waypoints,name);self.mode='execute';save(self.d/filename,q)
  if not q['valid']:raise ExecutionFailure('geometric_refusal:'+q['reason'])
 def execute(self):
  self.initialize();name='block_'+self.spec['target_color'];status='not_started';reason=None;failure_stage=None
  with physical_guard(pb,self.env,self.ctl.shadow,self.guard_audit):
   try:
    self.stage='preparation';self.ctl.phase='initial_settle';self.ctl.hold(24,1)
    self.mode='geometry';plans=self.ctl.plan_lift(name);self.mode='execute';_,waypoints,meta=plans[0];waypoints=[(np.array(pos,float),np.array(q,float))for pos,q in waypoints]
    if self.spec['geometry']=='offset_localY_12mm':
     shift=Rotation.from_quat(waypoints[2][1]).apply([0,.012,0]);waypoints=[(pos+(shift if i else 0),q)for i,(pos,q)in enumerate(waypoints)]
    save(self.d/'planning_private.json',{'all_candidates':self.ctl.planning_audit,'selected':meta,'waypoints':waypoints,'selected_index':0,'no_outcome_selection':True})
    self.checked(waypoints,name,'modified_path_check.json');(rise,q),(above,_),(grasp,_),(high,_)=waypoints
    self.ctl.move(rise,q,1,'raise_clearance');self.ctl.move(above,q,1,'above_named_block')
    if self.spec['preparation']=='empty_grasp_high':self.ctl.move(high,q,1,'direct_to_empty_high',speed=.0025,settle=18);self.ctl.hold(20,-1)
    else:
     self.ctl.move(grasp,q,1,'descend_named_block',speed=.0025);self.ctl.hold(20,-1)
     if self.spec['preparation']=='grasp_attempt_high':self.ctl.move(high,q,-1,'attempt_to_high',speed=.0025,settle=18)
    pre=self.capture('t0');obs=np.array(pre['robot_obs']);self.program=action_program(obs,self.spec['action']);self.pre_input=public_input(self.caps[0]['rgb'],self.caps[1]['rgb'],obs,self.program)
    npz(self.d/'precommitted_input_private.npz',**self.pre_input);save(self.d/'precommitted_action.json',{'commands':self.program,'seconds':2.4,'t0_tick':self.ticks,'t0_control':self.adapter.controls,'fixed_before_any_action_control':True})
    self.checked([(self.program[-1,:3],self.program[-1,3:7])],name,'action_path_check.json')
    if self.spec['action']=='hold_open':
     self.mode='geometry';check=opening_sweep(pb,self.ctl.shadow,name);self.mode='execute';save(self.d/'open_gripper_sweep.json',check)
     if not check['valid']:raise ExecutionFailure('geometric_refusal:'+check['reason'])
    self.stage='action';self.ctl.phase='preset_action'
    for i,cmd in enumerate(self.program,1):
     self.ctl.step(cmd[:3],cmd[3:7],int(cmd[7]))
     if i in ANCHORS:self.capture('t'+str(i))
    assert len(self.actual_actions)==72 and np.array_equal(np.array(self.actual_actions),self.program)
    uid=uuid.uuid5(uuid.NAMESPACE_URL,'wm54/'+self.rid).hex;pub=E/'data/model_inputs'/(uid+'.npz');npz(pub,**self.pre_input)
    self.public_path=str(pub);status='completed72'
   except ExecutionFailure as ex:
    reason=str(ex);failure_stage=self.stage;status='rejected_or_failed_preparation'if self.program is None else('rejected_action_before_execution'if not self.actual_actions else'aborted_action')
    self.mode='execute'
    if len(self.captures)<6:self.capture('failure_last_actual_state')
   finally:
    save(self.d/'all_planning_candidates_private.json',self.ctl.planning_audit);save(self.d/'actual_action_audit_private.json',{'actual_commands':self.actual_actions,'completed_controls':len(self.actual_actions),'actual_seconds':len(self.actual_actions)/30,'not_model_input_if_incomplete':len(self.actual_actions)!=72});save(self.d/'guard_audit.json',self.guard_audit);save(self.d/'safety.json',self.safety)
  summary={'root_id':self.rid,'split':self.spec['split'],'scene':self.spec['scene'],'target_color':self.spec['target_color'],'geometry':self.spec['geometry'],'preparation':self.spec['preparation'],'assigned_action':self.spec['action'],'execution_rank':self.spec['execution_rank'],'status':status,'reason':reason,'failure_stage':failure_stage,'controls':self.adapter.controls,'action_controls':len(self.actual_actions),'ticks':self.ticks,'initialization_ticks':self.initial_ticks,'states':len(self.captures),'camera_calls':self.camera_calls,'captures':self.captures,'model_input_path':getattr(self,'public_path',None),'safety':self.safety,'guard_audit':self.guard_audit,'one_initialization':True,'restores':0,'retries':0}
  assert self.ticks==self.initial_ticks+8*self.adapter.controls
  save(self.d/'root_summary.json',summary);return summary
 def close(self):
  for n,v in self.orig.items():setattr(pb,n,v)
  self.trace.close()
  if self.ctl is not None:self.ctl.close()
  if self.env is not None and self.env.ownsPhysicsClient:self.env.close();self.env.ownsPhysicsClient=False

def main():
 global HISTORICAL_RGB
 assert read(E/'results/preflight.json')['passed'];assert not(E/'results/attempts.json').exists()
 for p,h in read(E/'configs/precollection_frozen.json')['sha256'].items():assert sha(p)==h,p
 for p,h in read(E/'configs/input_references.json')['sha256'].items():assert sha(p)==h,p
 protocol=read(E/'configs/protocol.json');roots=sorted(protocol['roots'],key=lambda r:r['execution_rank']);HISTORICAL_RGB=set(read(E/'configs/historical_exclusion.json')['old_RGB_array_hashes']);install_write_guard()
 lock=(P/'runs/gpu.lock').open('r+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 save(E/'configs/runtime.json',{'utc':now(),'pid':os.getpid(),'python':sys.version,'platform':platform.platform(),'single_gpu_lock':True,'fits':0,'renderer':'native CALVIN EGL','runs_on':'local host'})
 summaries=[];attempts=[];started=time.perf_counter()
 try:
  for s in roots:
   attempts.append({'root_id':s['root_id'],'execution_rank':s['execution_rank'],'status':'started','utc':now()});save(E/'results/attempts.json',attempts);rt=None
   try:
    rt=Runtime(s);result=rt.execute();summaries.append(result);attempts[-1]['status']=result['status'];save(E/'results/root_summaries.json',summaries);save(E/'results/attempts.json',attempts)
    states=sum(r['states']for r in summaries)
    if states>=50 and not(E/'results/storage_first50_saved_states.json').exists():save(E/'results/storage_first50_saved_states.json',{'state_count_at_root_boundary':states,'data_bytes':sum(p.stat().st_size for p in(E/'data').rglob('*')if p.is_file()),'complete_transitions':sum(r['status']=='completed72'for r in summaries),'not_50_transition_claim':True})
    if sum(r['status']=='completed72'for r in summaries)>=50 and not(E/'results/storage_first50_transitions.json').exists():save(E/'results/storage_first50_transitions.json',{'complete_transitions':50,'data_bytes_including_prior_failures_at_boundary':sum(p.stat().st_size for p in(E/'data').rglob('*')if p.is_file())})
    print('ROOT',len(summaries),s['root_id'],result['status'],'action_controls',result['action_controls'],'states',result['states'],'reason',result['reason'],flush=True)
   except Exception:
    attempts[-1].update(status='infrastructure_or_lineage_integrity_block',traceback=traceback.format_exc());save(E/'results/attempts.json',attempts)
    if rt is not None:save(rt.d/'blocked_attempt.json',{'traceback':traceback.format_exc(),'controls':rt.adapter.controls if rt.adapter else 0,'ticks':rt.ticks,'captures':rt.captures,'action_controls':len(rt.actual_actions),'guard_audit':rt.guard_audit,'safety':rt.safety})
    raise
   finally:
    if rt is not None:rt.close()
 finally:
  save(E/'results/collection_summary.json',{'utc':now(),'assigned_roots':54,'attempted_roots':len(attempts),'root_summaries':len(summaries),'completed72':sum(r['status']=='completed72'for r in summaries),'states':sum(r['states']for r in summaries),'controls':sum(r['controls']for r in summaries),'ticks':sum(r['ticks']for r in summaries),'seconds':time.perf_counter()-started,'fits':0,'no_replacement':True,'data_bytes':sum(p.stat().st_size for p in(E/'data').rglob('*')if p.is_file()),'incomplete_attempt_counters':'see blocked_attempt.json if root_summaries<attempted_roots'})
  lock.close()
if __name__=='__main__':main()
