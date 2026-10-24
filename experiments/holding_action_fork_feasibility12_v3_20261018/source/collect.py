from common import *
import copy,time,sys,fcntl,traceback,math
from collections import deque
import pybullet as pb
import hydra
from hydra import initialize_config_dir,compose
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation
import labels_physical as truth
from wm_symbolic.physical_oracle_v2 import PhysicalOracleV2
from wm_symbolic.physical_oracle import ExecutionFailure
sys.path.insert(0,str(P/'scripts'))
from evaluate_physical_oracle_v2 import physics_guard

class ForkInvalid(RuntimeError):pass
SEEN_RGB_ROOTS={}
HISTORICAL_RGB=set()
def encoded(x):return json.dumps(x,sort_keys=True,default=json_default,separators=(',',':'))
def fingerprint(x):return hashlib.sha256(encoded(x).encode()).hexdigest()
def contact_multiset(x):
 # Only complete Bullet contact tuples are unordered. Preserve multiplicity,
 # every numeric field, and order of every other list, including histories.
 x=json.loads(encoded(x))
 def visit(v):
  if isinstance(v,dict):
   out={}
   for key,value in v.items():
    if key=='contacts':
     assert isinstance(value,list)and all(isinstance(c,list)and len(c)==14 for c in value)
     out[key]=sorted(value,key=encoded)
    else:out[key]=visit(value)
   return out
  if isinstance(v,list):return [visit(z)for z in v]
  return v
 return visit(x)
def compare(a,b):
 raw_equal=encoded(a)==encoded(b);a=contact_multiset(a);b=contact_multiset(b)
 differences=[];count=0;maximum=0.
 def walk(x,y,path):
  nonlocal count,maximum
  if isinstance(x,dict)and isinstance(y,dict):
   for k in sorted(set(x)|set(y)):
    if k not in x or k not in y:count+=1;differences.append({'path':path+'/'+str(k),'missing':True})
    else:walk(x[k],y[k],path+'/'+str(k))
  elif isinstance(x,list)and isinstance(y,list):
   if len(x)!=len(y):count+=1;differences.append({'path':path,'lengths':[len(x),len(y)]})
   for j,(v,w)in enumerate(zip(x,y)):walk(v,w,path+'/'+str(j))
  elif encoded(x)!=encoded(y):
   count+=1;delta=None
   if isinstance(x,(float,int))and isinstance(y,(float,int)):
    delta=abs(float(x)-float(y));maximum=max(maximum,delta)
   if len(differences)<100:differences.append({'path':path,'reference':x,'actual':y,'absolute_difference':delta})
 walk(a,b,'')
 return {'equal':fingerprint(a)==fingerprint(b),'raw_equal_before_contact_multiset':raw_equal,'reference_sha256':fingerprint(a),'actual_sha256':fingerprint(b),'different_scalars_or_structures':count,'maximum_absolute_numeric_difference_mixed_units':maximum,'first_differences':differences[:100]}

class Adapter:
 def __init__(self,rt):self.rt=rt;self.env=rt.env;self.controls=0;self.last_grip=1
 def observe(self):return {'robot_obs':self.env.get_state_obs()['robot_obs']}
 def step(self,action):
  if self.rt.condition_controls>=900:raise ExecutionFailure('condition_actual_control_budget_exhausted')
  self.env.robot.apply_action(action)
  for _ in range(self.env.action_repeat):pb.stepSimulation(physicsClientId=self.env.cid)
  self.env.scene.step();self.controls+=1;self.rt.total_controls+=1;self.rt.condition_controls+=1;self.last_grip=int(action[2]);return self.observe()

class Runtime:
 def __init__(self,spec,protocol,output_root=None,qualification=False):
  self.spec=spec;self.protocol=protocol;self.output=Path(output_root or E);self.qualification=qualification;self.actual_ticks=0;self.control_end_records={};self.first_tick_capture=None;self.spec=spec;self.rid=spec['root_id'];self.d=self.output/'data/private'/self.rid;self.d.mkdir(parents=True); self.env=None;self.ctl=None;self.mode='initialize';self.ticks=0;self.total_controls=0;self.condition_controls=0;self.hist=deque(maxlen=12);self.motors={};self.caps=[];self.snapshots=[];self.branch=None;self.local_step=None;self.prefix=[];self.records=[];self.branches=[];self.conditions=[];self.fidelity=[];self.prep_trace=[];self.tracefile=None;self.ref_monitor=None;self.safety_stats={};self.original={n:getattr(pb,n)for n in ['getCameraImage','stepSimulation','setJointMotorControl2']};pb.stepSimulation=self.step_sim;pb.getCameraImage=self.image;pb.setJointMotorControl2=self.motor
  with initialize_config_dir(config_dir=str(P/'vendor/calvin/calvin_env/conf'),job_name='fork_feasibility'):
   cfg=compose(config_name='config_data_collection',overrides=[f'scene=calvin_scene_{spec["scene"]}','cameras=static_and_gripper','use_vr=false',f'env.seed={spec["seed"]}','env.use_egl=true'])
  cfg.scene.objects.fixed_objects.table.joints.base__drawer.initial_state=spec['drawer_initial_q_m']
  for c,b in spec['blocks'].items():cfg.scene.objects.movable_objects['block_'+c].initial_pos=b['initial_pos'];cfg.scene.objects.movable_objects['block_'+c].initial_orn=b['initial_orn']
  OmegaConf.save(cfg,self.d/'resolved_config.yaml');self.env=hydra.utils.instantiate(cfg.env,use_scene_info=True);pb.setRealTimeSimulation(0,physicsClientId=self.env.cid);assert self.env.action_repeat==8;assert not self.env.robot.use_ik_fast
  saved=self.env.get_obs;self.env.get_obs=self.env.get_state_obs
  try:self.env.reset()
  finally:self.env.get_obs=saved
  self.adapter=Adapter(self);self.ctl=PhysicalOracleV2(self.adapter,self.record,max_controls=900);self.mapping={'robot_uid':self.env.robot.robot_uid,'object_uids':{c:self.ctl.objects['block_'+c].uid for c in COLORS},'finger_families':[sorted(f)for f in self.ctl.finger_families],'finger_joint_ids':list(self.env.robot.gripper_joint_ids),'table_uid':self.ctl.table.uid,'table_links':self.ctl.links};save(self.d/'body_link_mapping.json',self.mapping);self.mode='execute';self.tracefile=(self.d/'preparation_trace_private.jsonl').open('w');self.ctl.phase='initial_settle'
  with physics_guard(self.adapter,self.ctl):self.ctl.hold(24,1)
  self.base=self.checkpoint('base');self.condition_controls=0
 def motor(self,*a,**kw):
  cid=kw.get('physicsClientId',0);body=kw.get('bodyIndex',a[0]if a else None);joint=kw.get('jointIndex',a[1]if len(a)>1 else None);self.motors[f'{cid}:{body}:{joint}']={'args':copy.deepcopy(a),'kwargs':copy.deepcopy(kw)};return self.original['setJointMotorControl2'](*a,**kw)
 def step_sim(self,*a,**kw):
  assert self.mode in ['initialize','execute'],'physics advanced during capture/checkpoint'
  self.ticks+=1;self.actual_ticks+=1;out=self.original['stepSimulation'](*a,**kw)
  if self.branch is not None and self.local_step==1:
   n=len(self.branch_first_ticks)+1
   self.branch_first_ticks.append({'tick_within_first_control':n,'physical_full':self.physical(),'python_state':self.python_state(),'cutoffs':self.cutoff_labels()})
   if n==1:
    self.first_tick_capture=self.capture_native()
    if self.qualification:self.write_native(self.d/'branches'/self.branch/'first_physics_tick_images',self.first_tick_capture)
  return out
 def image(self,*a,**kw):
  assert self.mode in ['primary','baseline'],'unexpected RGB';kw['flags']=pb.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX if self.mode=='primary'else 0;im=self.original['getCameraImage'](*a,**kw);w,h=im[:2];self.caps.append({'rgb':np.asarray(im[2],np.uint8).reshape(h,w,4)[:,:,:3].copy(),'depth_buffer':np.asarray(im[3],np.float32).reshape(h,w).copy(),'mask':np.asarray(im[4],np.int32).reshape(h,w).copy(),'meta':{'tick':self.ticks,'viewMatrix':kw['viewMatrix'],'projectionMatrix':kw['projectionMatrix'],'width':w,'height':h,'flags':kw['flags']}});return im
 def physical(self):
  cid=self.env.cid;bodies=[]
  for k in range(pb.getNumBodies(physicsClientId=cid)):
   u=pb.getBodyUniqueId(k,physicsClientId=cid);n=pb.getNumJoints(u,physicsClientId=cid);bodies.append({'uid':u,'base_pose':pb.getBasePositionAndOrientation(u,physicsClientId=cid),'base_velocity':pb.getBaseVelocity(u,physicsClientId=cid),'joint_states':pb.getJointStates(u,list(range(n)),physicsClientId=cid)if n else [],'links':[pb.getLinkState(u,j,computeLinkVelocity=1,computeForwardKinematics=1,physicsClientId=cid)for j in range(n)],'dynamics':[pb.getDynamicsInfo(u,j,physicsClientId=cid)for j in range(-1,n)],'visual_shapes':pb.getVisualShapeData(u,physicsClientId=cid)})
  constraints=[]
  for k in range(pb.getNumConstraints(physicsClientId=cid)):
   uid=pb.getConstraintUniqueId(k,physicsClientId=cid);constraints.append({'id':uid,'info':pb.getConstraintInfo(uid,physicsClientId=cid),'applied_forces':pb.getConstraintState(uid,physicsClientId=cid)})
  contacts=sorted([list(c)for c in pb.getContactPoints(physicsClientId=cid)],key=encoded)
  return {'bodies':bodies,'constraints':constraints,'contacts':contacts,'engine_parameters':pb.getPhysicsEngineParameters(physicsClientId=cid)}
 def python_state(self):
  r=self.env.robot
  return {'robot':{k:copy.deepcopy(getattr(r,k))for k in ['gripper_action','prev_ee_orn','target_pos','target_orn','ll','ul','jr','rp']},'mixed_ik':{k:copy.deepcopy(getattr(r.mixed_ik,k))for k in ['is_using_IK_fast','ll','ul','jr','rp']},'scene':{group:[{k:copy.deepcopy(getattr(o,k))for k in ['state','prev_is_pressed']if hasattr(o,k)}for o in getattr(self.env.scene,group)]for group in ['buttons','switches','lights']},'rng':copy.deepcopy(self.env.np_random.bit_generator.state),'cameras':[{k:copy.deepcopy(getattr(cam,k))for k in ['view_matrix','projection_matrix','viewMatrix','projectionMatrix']if hasattr(cam,k)}for cam in self.env.cameras],'controller':{k:copy.deepcopy(getattr(self.ctl,k))for k in ['steps','phase','events','planning_audit','monitor','_plans']},'adapter':{'controls':self.adapter.controls,'last_grip':self.adapter.last_grip},'hist':copy.deepcopy(list(self.hist)),'ticks':self.ticks,'motors':copy.deepcopy(self.motors)}
 def restore_python(self,s):
  for k,v in s['robot'].items():setattr(self.env.robot,k,copy.deepcopy(v))
  for k,v in s['mixed_ik'].items():setattr(self.env.robot.mixed_ik,k,copy.deepcopy(v))
  for group,objects in s['scene'].items():
   for obj,values in zip(getattr(self.env.scene,group),objects):
    for k,v in values.items():setattr(obj,k,copy.deepcopy(v))
  self.env.np_random.bit_generator.state=copy.deepcopy(s['rng'])
  for cam,values in zip(self.env.cameras,s['cameras']):
   for k in ['view_matrix','projection_matrix','viewMatrix','projectionMatrix']:
    if k in values:setattr(cam,k,copy.deepcopy(values[k]))
    elif hasattr(cam,k):delattr(cam,k)
  for k,v in s['controller'].items():setattr(self.ctl,k,copy.deepcopy(v))
  self.adapter.controls=s['adapter']['controls'];self.adapter.last_grip=s['adapter']['last_grip'];self.hist=deque(copy.deepcopy(s['hist']),maxlen=12);self.ticks=s['ticks'];self.motors=copy.deepcopy(s['motors'])
  for call in s['motors'].values():
   if call['kwargs'].get('physicsClientId',0)==self.env.cid:self.original['setJointMotorControl2'](*call['args'],**call['kwargs'])
 def checkpoint(self,name):
  mode=self.mode;self.mode='checkpoint';physical=self.physical();python=self.python_state();state=pb.saveState(physicsClientId=self.env.cid);self.snapshots.append(state);pb.saveBullet(str(self.d/(name+'.bullet')),physicsClientId=self.env.cid);cp={'id':state,'physical':physical,'python':python};save(self.d/(name+'_checkpoint_private.json'),{'native_state_id':state,'physical':physical,'python':python,'opaque_solver_state':'native snapshot; not directly introspectable; observable state and shared12control prefix test required'});self.mode=mode;return cp
 def restore(self,cp,name):
  self.mode='checkpoint';pb.restoreState(stateId=cp['id'],physicsClientId=self.env.cid);native=self.physical();self.restore_python(cp['python']);actual=self.physical();a,ad=split_state(cp['physical']);b,bd=split_state(actual);cmp={'physical':compare(cp['physical'],actual),'physical_core':compare(a,b),'prior_reported_forces':compare(ad,bd),'motor_reissue_physical_effect':compare(native,actual),'python':compare(cp['python'],self.python_state())};save(self.d/(name+'_restore_actual_private.json'),actual);save(self.d/(name+'_native_before_motor_reissue_private.json'),native);save(self.d/(name+'_restore_comparison.json'),cmp);self.fidelity.append({'name':name,**cmp});self.mode='execute';return cmp
 def cutoff_labels(self):
  out=[]
  for threshold in [.005,.01,.015]:
   labels=[]
   for color in COLORS:
    cs=[c for c in self.ctl.contacts('block_'+color)if c[9]>threshold];robot=[c for c in cs if c[2]==self.env.robot.robot_uid];links={c[4]for c in robot};labels.append(bool(all(links&f for f in self.ctl.finger_families)and not any(c[2]!=self.env.robot.robot_uid for c in cs)))
   out.append(labels)
  return out
 def record(self,obs,action):
  physical=truth.snapshot(self.adapter,self.ctl);cutoffs=self.cutoff_labels();entry={'logical_control':self.adapter.controls,'tick':self.ticks,'actual_root_control_count':self.total_controls,'condition':getattr(self,'condition',None),'branch':self.branch,'branch_control':self.local_step,'action':action,'physical':physical,'holding_at_cutoffs':cutoffs};self.hist.append({'control':self.adapter.controls,'tick':self.ticks,'physical':physical,'holding_at_cutoffs':cutoffs});self.tracefile.write(encoded(entry)+'\n');self.tracefile.flush()
  if self.branch is not None and self.local_step is not None:
   item={'control':self.local_step,'action':action,'physical_full':self.physical(),'python_state':self.python_state(),'cutoffs':cutoffs,'history':list(self.hist)}
   self.control_end_records[self.local_step]=item
   if self.local_step<=12:self.prefix.append(item)
   if self.qualification:save(self.d/'branches'/self.branch/f'c{self.local_step:02d}.json',item)
  if self.ref_monitor is not None:
   robot=self.env.robot.robot_uid;target=self.ctl.objects['block_'+self.spec['target_color']].uid;contacts=[c for c in physical['contacts']if c[9]>.01 and robot in [c[1],c[2]]];bad=[c for c in contacts if(c[2]if c[1]==robot else c[1])!=target];penetration=max([max(0,-c[8])for c in contacts],default=0.);self.safety_stats['max_robot_penetration_m']=max(self.safety_stats.get('max_robot_penetration_m',0),penetration);self.safety_stats['unrequested_robot_contact_samples']=self.safety_stats.get('unrequested_robot_contact_samples',0)+len(bad)
   if penetration>=.01:raise ExecutionFailure('robot_penetration_at_least_10mm')
   if bad:raise ExecutionFailure('unrequested_robot_contact')
 def capture_native(self):
  self.caps=[];before=self.physical();phy=truth.snapshot(self.adapter,self.ctl);cuts=self.cutoff_labels();assert len(self.hist)==12
  stable=[all(all(h['holding_at_cutoffs'][k][j]==cuts[1][j]for k in range(3))and all(sum(x)<=1 for x in h['holding_at_cutoffs'])for h in self.hist)for j in range(3)]
  oldmode=self.mode;self.mode='primary';self.env.get_obs();self.mode='baseline'
  for cam in self.env.cameras:cam.render()
  self.mode='checkpoint';after=self.physical();assert encoded(before)==encoded(after)
  assert len(self.caps)==4 and len({c['meta']['tick']for c in self.caps})==1
  for a,b in zip(self.caps[:2],self.caps[2:]):assert np.array_equal(a['rgb'],b['rgb'])and np.array_equal(a['depth_buffer'],b['depth_buffer'])
  cap={'physical':before,'truth':phy,'cutoffs':cuts,'stable':stable,'history':list(self.hist),'caps':copy.deepcopy(self.caps[:2]),'python_after_capture':self.python_state(),'tick':self.ticks};self.mode=oldmode
  return cap
 def write_native(self,sd,cap):
  sd=Path(sd);sd.mkdir(parents=True,exist_ok=False)
  for cam,a in zip(['static','gripper'],cap['caps']):
   save_raw_npz(sd/f'native_{cam}.npz',rgb=a['rgb'],depth_buffer=a['depth_buffer'],mask=a['mask']);save(sd/f'camera_{cam}.json',a['meta'])
  save(sd/'integrity.json',{'tick':cap['tick'],'same_tick_both_cameras':True,'render_physics_unchanged':True,'baseline_RGB_depth_equal':True})
 def save_capture(self,bid,step,cap):
  sd=self.d/'branches'/bid/f's{step:02d}';self.write_native(sd,cap);save(sd/'physical_full.json',cap['physical']);save(sd/'truth_private.json',{'physical':cap['truth'],'cutoffs':cap['cutoffs'],'stable':cap['stable'],'last12_controls':cap['history']})
  pub=self.output/'data/public'/f'{bid}_s{step:02d}.npz';save_raw_npz(pub,rgb_static=cap['caps'][0]['rgb'],rgb_gripper=cap['caps'][1]['rgb']);rows=[]
  for cam,c in zip(['static','gripper'],cap['caps']):
   rgb_hash=hashlib.sha256(c['rgb'].tobytes()).hexdigest()
   if not self.qualification:
    assert rgb_hash not in HISTORICAL_RGB,'historical_RGB_duplicate';assert rgb_hash not in SEEN_RGB_ROOTS or SEEN_RGB_ROOTS[rgb_hash]==self.rid,'cross_root_RGB_duplicate';SEEN_RGB_ROOTS[rgb_hash]=self.rid
   seg=c['mask'];body=seg&((1<<24)-1);link=(seg>>24)-1;fingers=[int(((seg>=0)&(body==self.env.robot.robot_uid)&np.isin(link,list(f))).sum())for f in self.ctl.finger_families]
   for j,color in enumerate(COLORS):
    pixels=int(((seg>=0)&(body==self.mapping['object_uids'][color])).sum());visible=pixels>=4 and all(v>=4 for v in fingers);cs=[q for q in self.ctl.contacts('block_'+color)if q[9]>.01];support=[q for q in cs if q[2]==self.ctl.table.uid and q[4]==-1 and q[7][2]>.5]
    support_stable=all(bool([q for q in h['physical']['contacts']if ((q[1]==self.mapping['object_uids'][color]and q[2]==self.ctl.table.uid and q[4]==-1 and q[7][2]>.5)or(q[2]==self.mapping['object_uids'][color]and q[1]==self.ctl.table.uid and q[3]==-1 and q[7][2]<-.5))and q[9]>.01])==bool(support)for h in cap['history'])
    row={'root_id':self.rid,'condition':self.condition,'branch_id':bid,'action':self.branch_action,'control_index':step,'scene':self.spec['scene'],'target_color':self.spec['target_color'],'color':color,'camera':cam,'path':str(pub),'key':'rgb_'+cam,'RGB_sha256':rgb_hash,'Holding':bool(cap['cutoffs'][1][j]),'physical_valid':bool(cap['stable'][j]),'table_supported':bool(support),'table_support_last12_stable':support_stable,'table_upward_normal_force_N':sum(q[9]*q[7][2]for q in support),'visible':visible,'object_pixels':pixels,'finger_pixels':fingers,'unknown':not(cap['stable'][j]and visible),'lineage_split':'qualification_only'if self.qualification else'engineering_feasibility_only'};rows.append(row);self.records.append(row)
  cap['records']=rows;save(self.output/'results/records.json',self.records)
  if not self.qualification and len(list((self.output/'data/public').glob('*_s*.npz')))==50:save(self.output/'results/first50_storage.json',{'utc':now(),'saved_states':50,'public_bytes':sum(p.stat().st_size for p in(self.output/'data/public').rglob('*')if p.is_file()),'private_bytes':sum(p.stat().st_size for p in(self.output/'data/private').rglob('*')if p.is_file()),'not_50_independent_transitions':True})
  return rows
 def compare_captures(self,a,b,start=False):
  ac,ad=split_state(a['physical']);bc,bd=split_state(b['physical']);out={'physical':compare(a['physical'],b['physical']),'physical_core':compare(ac,bc),'prior_reported_forces':compare(ad,bd),'python':compare(a['python_after_capture'],b['python_after_capture']),'cutoffs':compare(a['cutoffs'],b['cutoffs']),'images':[]}
  for cam,ca,cb in zip(['static','gripper'],a['caps'],b['caps']):
   for field in ['rgb','depth_buffer','mask']:
    aa,bb=ca[field],cb[field];out['images'].append({'camera':cam,'field':field,'equal':bool(np.array_equal(aa,bb)),'unequal_values':int(np.sum(aa!=bb)),'reference_array_sha256':hashlib.sha256(aa.tobytes()).hexdigest(),'actual_array_sha256':hashlib.sha256(bb.tobytes()).hexdigest()})
  out['equal']=out['physical']['equal']and out['python']['equal']and out['cutoffs']['equal']and all(x['equal']for x in out['images']);out['equal_at_start']=out['physical_core']['equal']and out['python']['equal']and out['cutoffs']['equal']and all(x['equal']for x in out['images']);return out
 def set_monitor(self):
  self.ref_monitor={'scene':self.env.scene.get_obs().copy(),'objects':{n:self.ctl.xyz(n).copy()for n in self.ctl.objects},'allowed':{'block_'+self.spec['target_color']}};self.ctl.monitor=copy.deepcopy(self.ref_monitor);self.safety_stats={'max_robot_penetration_m':0.,'unrequested_robot_contact_samples':0}
 def prepare_condition(self,condition):
  self.condition=condition;self.condition_controls=0;self.branch=None;self.local_step=None;self.ref_monitor=None;summary={'condition':condition,'preparation':'started','branches':[]};self.conditions.append(summary)
  if self.tracefile:self.tracefile.close()
  self.tracefile=(self.d/(condition+'_preparation_trace_private.jsonl')).open('w');cmp=self.restore(self.base,condition+'_base')
  if not(cmp['physical_core']['equal']and cmp['python']['equal']and cmp['motor_reissue_physical_effect']['equal']):summary['preparation']='invalid_base_restore';raise ForkInvalid('base_restore_not_exact:'+condition)
  self.set_monitor();name='block_'+self.spec['target_color']
  try:
   with physics_guard(self.adapter,self.ctl):
    plans=self.ctl.plan_lift(name);_,original,meta=plans[0];waypoints=copy.deepcopy(original)
    if condition=='offset_12mm':
     delta=Rotation.from_quat(waypoints[2][1]).apply([0.,.012,0.])
     for j in [1,2,3]:waypoints[j]=(np.asarray(waypoints[j][0])+delta,waypoints[j][1])
    check=self.ctl.shadow.check(waypoints,name);save(self.d/(condition+'_plan_private.json'),{'candidate':meta,'all_candidates':self.ctl.planning_audit,'condition_waypoints':waypoints,'condition_clearance':check})
    if not check['valid']:raise ExecutionFailure('condition_clearance_rejected:'+check['reason'])
    self.ctl.move(*waypoints[0],1,'raise_clearance');self.ctl.move(*waypoints[1],1,'above_named_block');self.ctl.move(*waypoints[2],1,'descend_named_block',speed=.0025);self.ctl.phase='prepare_closed';self.ctl.hold(20,-1)
   basepos,quat=pb.getLinkState(self.env.robot.robot_uid,self.env.robot.tcp_link_id,physicsClientId=self.env.cid)[:2];basepos=np.asarray(basepos);quat=np.asarray(quat);check=self.ctl.shadow.check([(basepos,quat),(basepos+[0,0,.12],quat)],name);save(self.d/(condition+'_branch_clearance.json'),check)
   if not check['valid']:raise ExecutionFailure('branch_clearance_rejected:'+check['reason'])
   summary['preparation']='safe_completed';summary['preparation_safety']=copy.deepcopy(self.safety_stats);self.set_monitor();cp=self.checkpoint(condition+'_live_action_start')
   return {'cp':cp,'origin':basepos,'quaternion':quat,'summary':summary}
  except ExecutionFailure as exc:
   summary['preparation']='failed';summary['reason']=str(exc);summary['safety']=copy.deepcopy(self.safety_stats);save(self.d/(condition+'_failed_planning_private.json'),{'reason':str(exc),'all_candidates':self.ctl.planning_audit,'events':self.ctl.events,'actual_controls':self.condition_controls});save(self.d/'conditions.json',self.conditions);return None
 def execute_branch(self,action,bid,prepared,reference=None,compare_steps=12):
  # The only branch implementation: the first branch ALSO restores live S.
  self.branch=bid;self.branch_action=action;self.local_step=0;self.prefix=[];self.branch_first_ticks=[];self.control_end_records={};self.first_tick_capture=None
  bd=self.d/'branches'/bid;bd.mkdir(parents=True);brow={'root_id':self.rid,'condition':self.condition,'action':action,'branch_id':bid,'status':'started','saved_states':0,'executed_controls':0,'fork_valid':False,'symmetric_live_restore':True};self.branches.append(brow);prepared['summary']['branches'].append(brow)
  if self.tracefile:self.tracefile.close()
  self.tracefile=(bd/'control_trace_private.jsonl').open('w');checks=[];commands=[];captures={};branch_start=self.total_controls;pre=None
  try:
   cmp=self.restore(prepared['cp'],bid+'_start');brow['restore']=cmp
   if not(cmp['physical_core']['equal']and cmp['python']['equal']and cmp['motor_reissue_physical_effect']['equal']):raise ForkInvalid('live_start_restore_mismatch')
   self.set_monitor();pre=self.capture_native();self.save_capture(bid,0,pre);brow['saved_states']+=1;brow['initial_records']=pre['records'];save(bd/'c00.json',{'physical_full':pre['physical'],'python_state':pre['python_after_capture'],'cutoffs':pre['cutoffs']})
   if reference is not None:
    cmp0=self.compare_captures(reference['pre'],pre,start=True);checks.append({'where':'start0','capture':cmp0,'equal':cmp0['equal_at_start']})
    if not cmp0['equal_at_start']:raise ForkInvalid('matched_start_core_Python_or_images_mismatch')
   save(bd/'action_program.json',{'semantic_action':action,'relative_program':'hold12 then dz ramp0.12 over48 then settle12'if action=='lift'else'hold72','target_origin':prepared['origin'],'quaternion':prepared['quaternion'],'controls':72,'duration_seconds':2.4,'tick_repeat':8})
   with physics_guard(self.adapter,self.ctl):
    for step in range(1,73):
     self.local_step=step;delta=0. if action=='hold'or step<=12 else .12*min(1.,(step-12)/48);xyz=prepared['origin']+np.array([0,0,delta]);quat=prepared['quaternion'];self.ctl.phase='branch_shared_prefix'if step<=12 else'branch_'+action;commands.append([*xyz.tolist(),*quat.tolist(),-1]);self.ctl.step(xyz,quat,-1);brow['executed_controls']=step
     # Identical capture calls in qualification and collection. Only persistence
     # differs: qualification saves all; collection saves the original6anchors.
     cap=self.capture_native()
     if step<=12 or self.qualification:captures[step]=cap
     if self.qualification or step in self.protocol['budget']['saved_control_indices']:
      self.save_capture(bid,step,cap);brow['saved_states']+=1
      if step==12:brow['matched_action_start_records']=cap['records']
      if step==72:brow['final_records']=cap['records']
     if reference is not None and step<=compare_steps:
      tr=compare(reference['trace'][step],self.control_end_records[step]);im=self.compare_captures(reference['captures'][step],cap);item={'where':f'control{step:02d}','trace':tr,'capture':im,'equal':tr['equal']and im['equal']};checks.append(item)
      if step==1:
       ticks=compare(reference['first_ticks'],self.branch_first_ticks);firstim=self.compare_captures(reference['first_tick_capture'],self.first_tick_capture);checks.append({'where':'first_control_8physics_ticks','trace':ticks,'first_tick_capture':firstim,'equal':ticks['equal']and firstim['equal']})
      if not all(c['equal']for c in checks):raise ForkInvalid('matched_replay_or_common_prefix_mismatch:'+str(step))
   brow['status']='completed'
  except ExecutionFailure as exc:brow['status']='failed';brow['reason']=str(exc)
  except ForkInvalid as exc:brow['status']='invalid_fork';brow['reason']=str(exc);raise
  finally:
   brow['executed_controls']=self.total_controls-branch_start;brow['safety']=copy.deepcopy(self.safety_stats);brow['compared_controls']=sum(c['where'].startswith('control')for c in checks);brow['checks_passed']=all(c['equal']for c in checks);brow['reference_branch']=reference['row']['branch_id']if reference else None
   save(bd/'common_prefix_trace_private.json',self.prefix);save(bd/'first_control_8physics_ticks_private.json',self.branch_first_ticks);save(bd/'fidelity_checks.json',checks);save(self.output/'data/public'/f'{bid}_actions.json',{'action_vectors_xyz_quaternion_grip':commands,'issued_controls':len(commands),'completed_controls':brow['executed_controls'],'physics_ticks_per_control':8,'planned_duration_controls':72});save(bd/'branch_result.json',brow);save(self.output/'results/current_branches.json',self.branches)
  result={'row':brow,'pre':pre,'trace':copy.deepcopy(self.control_end_records),'captures':captures,'first_ticks':copy.deepcopy(self.branch_first_ticks),'first_tick_capture':self.first_tick_capture,'checks':checks};self.branch=None;self.local_step=None;return result
 def execute_condition(self,condition):
  prepared=self.prepare_condition(condition)
  if prepared is None:return
  reference=None;shared=False
  for action in self.spec['action_order']:
   bid=f'{self.rid}_{condition}_{action}';result=self.execute_branch(action,bid,prepared,reference,12)
   if reference is None:
    reference=result
    if result['row']['executed_controls']<12:prepared['summary']['reason']='unsafe_before_common_prefix_complete';break
   else:shared=result['row']['compared_controls']>=12 and result['row']['checks_passed']
  prepared['summary']['shared12_prefix_verified']=shared
  for b in prepared['summary']['branches']:b['fork_valid']=bool(shared and len(prepared['summary']['branches'])==2);save(self.d/'branches'/b['branch_id']/'branch_result.json',b)
  save(self.d/'conditions.json',self.conditions);self.branch=None;self.local_step=None;self.ref_monitor=None
 def run(self):
  for condition in self.spec['condition_order']:self.execute_condition(condition)
  return {'root_id':self.rid,'target_color':self.spec['target_color'],'scene':self.spec['scene'],'status':'completed_all_prespecified_attempts','conditions':self.conditions,'branches':self.branches,'saved_states':sum(b['saved_states']for b in self.branches),'actual_controls':self.total_controls,'actual_physics_ticks':self.actual_ticks,'fidelity':self.fidelity}
 def close(self):
  if self.tracefile:self.tracefile.close()
  for name,value in self.original.items():setattr(pb,name,value)
  if self.ctl is not None:self.ctl.close()
  if self.env is not None:
   for state in self.snapshots:
    try:pb.removeState(state,physicsClientId=self.env.cid)
    except Exception:pass
   self.env.close()

def main():
 global HISTORICAL_RGB
 global_records=[];allbranches=[];results=[];attempts=[];assert not(E/'results/attempts.json').exists();protocol=read(E/'configs/protocol.json')
 assert read(Q/'results/QUALIFICATION.json')['passed'],'live symmetric full72 qualification is required before collection'
 HISTORICAL_RGB=set(read(E/'configs/historical_exclusion.json')['old_RGB_array_hashes'])
 for p,h in read(E/'configs/frozen.json')['sha256'].items():assert sha(p)==h,p
 for p,h in read(E/'configs/input_references.json')['sha256'].items():assert sha(p)==h,p
 lock=(P/'runs/gpu.lock').open('r+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.perf_counter();stop=None
 for spec in protocol['roots']:
  attempts.append({'root_id':spec['root_id'],'status':'started','utc':now()});save(E/'results/attempts.json',attempts);rt=None
  try:
   rt=Runtime.__new__(Runtime);rt.__init__(spec,protocol);result=rt.run();results.append(result);attempts[-1]['status']='completed'
  except ForkInvalid as exc:
   stop={'kind':'exact_fork_gate_failure','root_id':spec['root_id'],'reason':str(exc),'utc':now(),'interpret_action_differences':False};attempts[-1]['status']='stopped_invalid_fork';save(E/'results/STOPPED.json',stop)
   if rt:results.append({'root_id':rt.rid,'target_color':spec['target_color'],'scene':spec['scene'],'status':'invalid_fork','conditions':rt.conditions,'branches':rt.branches,'saved_states':sum(b['saved_states']for b in rt.branches),'actual_controls':rt.total_controls,'actual_physics_ticks':rt.actual_ticks,'fidelity':rt.fidelity})
  except Exception as exc:
   stop={'kind':'execution_error','root_id':spec['root_id'],'exception':repr(exc),'traceback':traceback.format_exc(),'utc':now(),'interpret_action_differences':False};attempts[-1]['status']='execution_error';save(E/'results/STOPPED.json',stop)
  finally:
   if rt:
    global_records.extend(rt.records);allbranches.extend(rt.branches);rt.close()
   save(E/'results/records.json',global_records);save(E/'results/branches.json',allbranches);save(E/'results/root_results.json',results);save(E/'results/attempts.json',attempts)
  print('ROOT_RESULT',spec['root_id'],attempts[-1]['status'],'savedstates',sum(b['saved_states']for b in allbranches),flush=True)
  if stop:break
 states=sum(b['saved_states']for b in allbranches);assert len(attempts)<=12 and len(allbranches)<=48 and states<=288 and all(b['saved_states']<=6 for b in allbranches)
 save(E/'results/collection_summary.json',{'utc':now(),'roots_planned':12,'roots_attempted':len(attempts),'branches_attempted':len(allbranches),'states_saved':states,'primary_images':states*2,'seconds':time.perf_counter()-start,'stop':stop,'fits':0,'action_differences_interpretable':stop is None and any(b['fork_valid']for b in allbranches),'remaining_roots_not_attempted':[r['root_id']for r in protocol['roots'][len(attempts):]]});lock.close();print('PILOT_END',stop,flush=True)
if __name__=='__main__':main()
