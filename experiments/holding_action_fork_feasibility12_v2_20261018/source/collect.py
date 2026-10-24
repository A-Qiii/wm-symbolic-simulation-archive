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
def compare(a,b):
 differences=[];count=0;maximum=0.
 def walk(x,y,path):
  nonlocal count,maximum
  if isinstance(x,dict)and isinstance(y,dict):
   for k in sorted(set(x)|set(y)):
    if k not in x or k not in y:count+=1;differences.append({'path':path+'/'+str(k),'missing':True})
    else:walk(x[k],y[k],path+'/'+str(k))
  elif isinstance(x,(list,tuple))and isinstance(y,(list,tuple)):
   if len(x)!=len(y):count+=1;differences.append({'path':path,'lengths':[len(x),len(y)]})
   for j,(v,w)in enumerate(zip(x,y)):walk(v,w,path+'/'+str(j))
  elif encoded(x)!=encoded(y):
   count+=1;delta=None
   if isinstance(x,(float,int,np.number))and isinstance(y,(float,int,np.number)):
    delta=abs(float(x)-float(y));maximum=max(maximum,delta)
   if len(differences)<100:differences.append({'path':path,'reference':x,'actual':y,'absolute_difference':delta})
 walk(json.loads(encoded(a)),json.loads(encoded(b)),'')
 return {'equal':fingerprint(a)==fingerprint(b),'reference_sha256':fingerprint(a),'actual_sha256':fingerprint(b),'different_scalars_or_structures':count,'maximum_absolute_numeric_difference_mixed_units':maximum,'first_differences':differences[:100]}

class Adapter:
 def __init__(self,rt):self.rt=rt;self.env=rt.env;self.controls=0;self.last_grip=1
 def observe(self):return {'robot_obs':self.env.get_state_obs()['robot_obs']}
 def step(self,action):
  if self.rt.condition_controls>=900:raise ExecutionFailure('condition_actual_control_budget_exhausted')
  self.env.robot.apply_action(action)
  for _ in range(self.env.action_repeat):pb.stepSimulation(physicsClientId=self.env.cid)
  self.env.scene.step();self.controls+=1;self.rt.total_controls+=1;self.rt.condition_controls+=1;self.last_grip=int(action[2]);return self.observe()

class Runtime:
 def __init__(self,spec,protocol):
  self.spec=spec;self.protocol=protocol;self.rid=spec['root_id'];self.d=E/'data/private'/self.rid;self.d.mkdir();self.env=None;self.ctl=None;self.mode='initialize';self.ticks=0;self.total_controls=0;self.condition_controls=0;self.hist=deque(maxlen=12);self.motors={};self.caps=[];self.snapshots=[];self.branch=None;self.local_step=None;self.prefix=[];self.records=[];self.branches=[];self.conditions=[];self.fidelity=[];self.prep_trace=[];self.tracefile=None;self.ref_monitor=None;self.safety_stats={};self.original={n:getattr(pb,n)for n in ['getCameraImage','stepSimulation','setJointMotorControl2']};pb.stepSimulation=self.step_sim;pb.getCameraImage=self.image;pb.setJointMotorControl2=self.motor
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
  assert self.mode in ['initialize','execute'],'physics advanced during capture/checkpoint';self.ticks+=1;out=self.original['stepSimulation'](*a,**kw)
  if self.branch is not None and self.local_step==1:
   self.branch_first_ticks.append({'tick_within_first_control':len(self.branch_first_ticks)+1,'physical_full':self.physical(),'python_state':self.python_state()})
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
  if self.branch is not None and self.local_step is not None and self.local_step<=12:self.prefix.append({'control':self.local_step,'action':action,'physical_full':self.physical(),'python_state':self.python_state()})
  if self.ref_monitor is not None:
   robot=self.env.robot.robot_uid;target=self.ctl.objects['block_'+self.spec['target_color']].uid;contacts=[c for c in physical['contacts']if c[9]>.01 and robot in [c[1],c[2]]];bad=[c for c in contacts if(c[2]if c[1]==robot else c[1])!=target];penetration=max([max(0,-c[8])for c in contacts],default=0.);self.safety_stats['max_robot_penetration_m']=max(self.safety_stats.get('max_robot_penetration_m',0),penetration);self.safety_stats['unrequested_robot_contact_samples']=self.safety_stats.get('unrequested_robot_contact_samples',0)+len(bad)
   if penetration>=.01:raise ExecutionFailure('robot_penetration_at_least_10mm')
   if bad:raise ExecutionFailure('unrequested_robot_contact')
 def capture(self,bid,step):
  sd=self.d/'branches'/bid/f's{step:02d}';sd.mkdir(parents=True,exist_ok=False);self.caps=[];before=self.physical();phy=truth.snapshot(self.adapter,self.ctl);cuts=self.cutoff_labels();assert len(self.hist)==12;stable=[all(all(h['holding_at_cutoffs'][k][j]==cuts[1][j]for k in range(3))and all(sum(x)<=1 for x in h['holding_at_cutoffs'])for h in self.hist)for j in range(3)];self.mode='primary';self.env.get_obs();self.mode='baseline'
  for cam in self.env.cameras:cam.render()
  self.mode='checkpoint';after=self.physical();assert compare(before,after)['equal'];assert len(self.caps)==4 and len({c['meta']['tick']for c in self.caps})==1
  for a,b in zip(self.caps[:2],self.caps[2:]):assert np.array_equal(a['rgb'],b['rgb'])and np.array_equal(a['depth_buffer'],b['depth_buffer'])
  pub=E/'data/public'/f'{bid}_s{step:02d}.npz';save_raw_npz(pub,rgb_static=self.caps[0]['rgb'],rgb_gripper=self.caps[1]['rgb']);save(sd/'physical_full.json',before);save(sd/'truth_private.json',{'physical':phy,'cutoffs':cuts,'stable':stable,'last12_controls':list(self.hist)});state_rows=[]
  for cam,cap in zip(self.env.cameras,self.caps[:2]):
   rgb_hash=hashlib.sha256(cap['rgb'].tobytes()).hexdigest();assert rgb_hash not in HISTORICAL_RGB,'historical_RGB_duplicate';assert rgb_hash not in SEEN_RGB_ROOTS or SEEN_RGB_ROOTS[rgb_hash]==self.rid,'cross_root_RGB_duplicate';SEEN_RGB_ROOTS[rgb_hash]=self.rid
   seg=cap['mask'];body=seg&((1<<24)-1);link=(seg>>24)-1;finger_pixels=[int(((seg>=0)&(body==self.env.robot.robot_uid)&np.isin(link,list(f))).sum())for f in self.ctl.finger_families];save_raw_npz(sd/f'native_{cam.name}.npz',rgb=cap['rgb'],depth_buffer=cap['depth_buffer'],mask=seg);save(sd/f'camera_{cam.name}.json',cap['meta'])
   for j,c in enumerate(COLORS):
    pixels=int(((seg>=0)&(body==self.mapping['object_uids'][c])).sum());visible=pixels>=4 and all(v>=4 for v in finger_pixels);cs=[q for q in self.ctl.contacts('block_'+c)if q[9]>.01];support=[q for q in cs if q[2]==self.ctl.table.uid and q[4]==-1 and q[7][2]>.5];row={'root_id':self.rid,'condition':self.condition,'branch_id':bid,'action':self.branch_action,'control_index':step,'scene':self.spec['scene'],'target_color':self.spec['target_color'],'color':c,'camera':cam.name,'path':str(pub),'key':'rgb_'+cam.name,'RGB_sha256':hashlib.sha256(cap['rgb'].tobytes()).hexdigest(),'Holding':bool(cuts[1][j]),'physical_valid':bool(stable[j]),'table_supported':bool(support),'table_upward_normal_force_N':sum(q[9]*q[7][2]for q in support),'visible':visible,'object_pixels':pixels,'finger_pixels':finger_pixels,'unknown':not(stable[j]and visible),'lineage_split':'engineering_feasibility_only'};state_rows.append(row);self.records.append(row)
  self.mode='execute';result={'physical':before,'caps':copy.deepcopy(self.caps[:2]),'records':state_rows,'public_path':str(pub),'step':step};save(sd/'integrity.json',{'tick':self.ticks,'same_tick_both_cameras':True,'render_physics_unchanged':True,'baseline_RGB_depth_equal':True});save(E/'results/records.json',self.records)
  if len(list((E/'data/public').glob('*_s*.npz')))==50:
   save(E/'results/first50_storage.json',{'utc':now(),'saved_states':50,'public_bytes':sum(p.stat().st_size for p in(E/'data/public').rglob('*')if p.is_file()),'private_bytes_including_native_checkpoints':sum(p.stat().st_size for p in(E/'data/private').rglob('*')if p.is_file()),'not_50_independent_transitions':True})
  return result
 def compare_captures(self,a,b):
  ac,ad=split_state(a['physical']);bc,bd=split_state(b['physical']);out={'physical':compare(a['physical'],b['physical']),'physical_core':compare(ac,bc),'prior_reported_forces':compare(ad,bd),'images':[]}
  for cam,ca,cb in zip(['static','gripper'],a['caps'],b['caps']):
   for field in ['rgb','depth_buffer','mask']:
    aa,bb=ca[field],cb[field];out['images'].append({'camera':cam,'field':field,'equal':np.array_equal(aa,bb),'unequal_values':int(np.sum(aa!=bb)),'max_absolute_difference':float(np.max(np.abs(aa.astype(float)-bb.astype(float))))})
  out['equal']=out['physical']['equal']and all(x['equal']for x in out['images']);out['equal_core_and_images']=out['physical_core']['equal']and all(x['equal']for x in out['images']);return out
 def set_monitor(self):
  self.ref_monitor={'scene':self.env.scene.get_obs().copy(),'objects':{n:self.ctl.xyz(n).copy()for n in self.ctl.objects},'allowed':{'block_'+self.spec['target_color']}};self.ctl.monitor=copy.deepcopy(self.ref_monitor);self.safety_stats={'max_robot_penetration_m':0.,'unrequested_robot_contact_samples':0}
 def execute_condition(self,condition):
  self.condition=condition;self.condition_controls=0;self.branch=None;self.local_step=None;self.ref_monitor=None;summary={'condition':condition,'preparation':'started','branches':[]};self.conditions.append(summary)
  if self.tracefile:self.tracefile.close()
  self.tracefile=(self.d/(condition+'_preparation_trace_private.jsonl')).open('w')
  cmp=self.restore(self.base,condition+'_base')
  if not(cmp['physical_core']['equal']and cmp['python']['equal']and cmp['motor_reissue_physical_effect']['equal']):
   summary['preparation']='invalid_base_restore';raise ForkInvalid('base_restore_not_exact:'+condition)
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
   summary['preparation']='safe_completed';summary['preparation_safety']=copy.deepcopy(self.safety_stats)
  except ExecutionFailure as exc:
   summary['preparation']='failed';summary['reason']=str(exc);summary['safety']=copy.deepcopy(self.safety_stats);save(self.d/(condition+'_failed_planning_private.json'),{'reason':str(exc),'all_candidates':self.ctl.planning_audit,'events':self.ctl.events,'actual_controls':self.condition_controls});save(self.d/'conditions.json',self.conditions);return
  cp=None;pre_reference=None;prefix_reference=None;prefix_capture=None;tick_reference=None;shared_verified=False
  for ai,action in enumerate(self.spec['action_order']):
   bid=f'{self.rid}_{condition}_{action}';self.branch=bid;self.branch_action=action;self.local_step=0;self.prefix=[];self.branch_first_ticks=[];bd=self.d/'branches'/bid;bd.mkdir(parents=True,exist_ok=True);brow={'root_id':self.rid,'condition':condition,'action':action,'branch_id':bid,'status':'started','saved_states':0,'executed_controls':0,'fork_valid':False};self.branches.append(brow);summary['branches'].append(brow);save(E/'results/current_branches.json',self.branches)
   if self.tracefile:self.tracefile.close()
   self.tracefile=(bd/'control_trace_private.jsonl').open('w')
   if ai:
    cmp=self.restore(cp,condition+'_'+action+'_start');pre=self.capture(bid,0);brow['saved_states']+=1;precmp=self.compare_captures(pre_reference,pre);save(bd/'pre_action_fidelity.json',precmp)
    if not(cmp['physical_core']['equal']and cmp['python']['equal']and cmp['motor_reissue_physical_effect']['equal']and precmp['equal_core_and_images']):brow['status']='invalid_fork';brow['reason']='pre_action_restore_mismatch';save(bd/'branch_result.json',brow);raise ForkInvalid(brow['reason'])
   else:
    pre=self.capture(bid,0);brow['saved_states']+=1;pre_reference=pre;cp=self.checkpoint(condition+'_action_start')
   self.set_monitor();commands=[];branch_start_actual_controls=self.total_controls;self.safety_stats={'max_robot_penetration_m':0.,'unrequested_robot_contact_samples':0};save(bd/'action_program.json',{'semantic_action':action,'relative_program':'hold12 then dz ramp0.12 over48 then settle12'if action=='lift'else'hold72','target_origin':basepos,'quaternion':quat,'controls':72,'duration_seconds':2.4,'tick_repeat':8})
   try:
    with physics_guard(self.adapter,self.ctl):
     for step in range(1,73):
      self.local_step=step;delta=0. if action=='hold'or step<=12 else .12*min(1.,(step-12)/48);xyz=basepos+np.array([0,0,delta]);self.ctl.phase='branch_shared_prefix'if step<=12 else'branch_'+action;commands.append([*xyz.tolist(),*quat.tolist(),-1]);self.ctl.step(xyz,quat,-1);brow['executed_controls']=step
      if step in self.protocol['budget']['saved_control_indices']:
       cap=self.capture(bid,step);brow['saved_states']+=1
       if step==12:
        brow['matched_action_start_records']=cap['records']
        if ai:
         cmp1=compare(prefix_reference,self.prefix);cmp2=self.compare_captures(prefix_capture,cap);cmp3=compare(tick_reference,self.branch_first_ticks);save(bd/'shared_prefix_fidelity.json',{'controls':12,'trace':cmp1,'capture':cmp2,'first_control_8physics_ticks':cmp3});shared_verified=cmp1['equal']and cmp2['equal']and cmp3['equal']
         if not shared_verified:raise ForkInvalid('identical12control_prefix_not_exact')
        else:prefix_reference=copy.deepcopy(self.prefix);prefix_capture=cap;tick_reference=copy.deepcopy(self.branch_first_ticks)
    brow['status']='completed';brow['final_records']=cap['records'];brow['safety']=copy.deepcopy(self.safety_stats)
   except ExecutionFailure as exc:brow['status']='failed';brow['reason']=str(exc);brow['safety']=copy.deepcopy(self.safety_stats)
   except ForkInvalid as exc:
    brow['status']='invalid_fork';brow['reason']=str(exc);brow['safety']=copy.deepcopy(self.safety_stats);raise
   finally:
    brow['executed_controls']=self.total_controls-branch_start_actual_controls
    save(bd/'common_prefix_trace_private.json',self.prefix);save(bd/'first_control_8physics_ticks_private.json',self.branch_first_ticks)
    public=E/'data/public'/f'{bid}_actions.json';save(public,{'action_vectors_xyz_quaternion_grip':commands,'issued_controls':len(commands),'completed_controls':brow['executed_controls'],'physics_ticks_per_control':8,'planned_duration_controls':72,'planned_duration_seconds':2.4});save(bd/'branch_result.json',brow);save(E/'results/current_branches.json',self.branches)
   if prefix_reference is None:summary['reason']='unsafe_before_common_prefix_complete';break
  summary['shared12_prefix_verified']=shared_verified
  for b in summary['branches']:b['fork_valid']=bool(shared_verified and len(summary['branches'])==2);save(self.d/'branches'/b['branch_id']/'branch_result.json',b)
  save(self.d/'conditions.json',self.conditions);self.branch=None;self.local_step=None;self.ref_monitor=None
 def run(self):
  for condition in self.spec['condition_order']:self.execute_condition(condition)
  return {'root_id':self.rid,'target_color':self.spec['target_color'],'scene':self.spec['scene'],'status':'completed_all_prespecified_attempts','conditions':self.conditions,'branches':self.branches,'saved_states':sum(b['saved_states']for b in self.branches),'actual_controls':self.total_controls,'fidelity':self.fidelity}
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
 assert read(Q/'results/QUALIFICATION.json')['passed'],'full72history qualification is required before collection'
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
   if rt:results.append({'root_id':rt.rid,'target_color':spec['target_color'],'scene':spec['scene'],'status':'invalid_fork','conditions':rt.conditions,'branches':rt.branches,'saved_states':sum(b['saved_states']for b in rt.branches),'actual_controls':rt.total_controls,'fidelity':rt.fidelity})
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
