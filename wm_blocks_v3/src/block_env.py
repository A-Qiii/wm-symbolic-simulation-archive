"""Blocks experiment. Full robot/object collisions; attachment requires bilateral contact.
Private control and scoring may use simulation geometry; learned deployment cannot.
"""
from paths import CALVIN_ENV
from paths import ROOT as B, RUNTIME as P
from pathlib import Path
from collections import deque
import random,json,copy,math
import numpy as np,pybullet as p,hydra
from hydra import initialize_config_dir,compose
from omegaconf import OmegaConf
from calvin_adapter import CalvinAdapter
from kinematic_guard import Guard

SIZE={'large':np.array([.10,.05,.05]),'small':np.array([.02,.02,.02])}
def serial(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 raise TypeError(type(x).__name__)
def save(path,x):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=serial)+'\n')
def region(pos):
 x,y,z=pos
 if not(-.35<x<.4 and -.3<y<.15 and z>.4):return 2
 return 0 if x<.075 else 1 if x>.115 else 2

class Scene:
 def __init__(self,layout,capture=False,assist=True):
  self.layout=layout;self.cid=None;self.guard=None;self.assist=assist;self.held=None;self.constraint=None;self.offset=None;self.trace=[];self.frames=[];self.phases=[];self.stage='initialization';self.capture=capture;self.grasp_audits=[];self.steps=0
  random.seed(layout['seed']);np.random.seed(layout['seed'])
  with initialize_config_dir(config_dir=str(CALVIN_ENV/'conf'),job_name='block_contact'):
   c=compose(config_name='config_data_collection',overrides=['scene=calvin_scene_A','robot=panda','cameras=static_and_gripper','use_vr=false','env.use_egl=true',f'env.seed={layout["seed"]}','env.bullet_time_step=960'])
  xy=np.array(layout['large_xy']);yaw=layout.get('yaw',0.);R=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]]);small=np.array(layout.get('small_xy',xy+R@np.array(layout.get('small_relative_xy',[.03,0.]))if layout['stacked']else xy+np.array([.12,0.])));zs=.5003 if layout['stacked']else .4503
  c.scene.objects.movable_objects=OmegaConf.create({n:{'file':str(B/'assets'/f'{n}.urdf'),'initial_pos':[float(x[0]),float(x[1]),z],'initial_orn':[0,0,yaw if n=='large'else 0]}for n,x,z in [('large',xy,.4653),('small',small,zs)]})
  ad=CalvinAdapter.__new__(CalvinAdapter);ad.project=P;ad.environment='A';ad.config=c;ad.env=hydra.utils.instantiate(c.env,use_scene_info=True);ad.with_segmentation=False;ad._original_camera_call=None;ad.history=deque(maxlen=4)
  self.ad=ad;self.cid=ad.env.cid;self.robot=ad.env.robot;self.uid={o.name:o.uid for o in ad.env.scene.movable_objects};self.names={self.robot.robot_uid:'robot',**{u:n for n,u in self.uid.items()}}
  for o in ad.env.scene.fixed_objects:self.names[o.uid]='fixed:'+o.name
  self.fingers=list(self.robot.gripper_joint_ids)
  assert len(self.fingers)==2
  # Slow, force-limited closure. CALVIN's default 200 N / 1 m/s commands
  # are unsuitable for these small rigid boxes and produce transient overlap.
  def control_gripper(grip):
   for link in self.fingers:p.setJointMotorControl2(self.robot.robot_uid,link,p.POSITION_CONTROL,targetPosition=.04 if grip==1 else 0.,force=15.,maxVelocity=.025,positionGain=.05,velocityGain=1.,physicsClientId=self.cid)
  self.robot.control_gripper=control_gripper
  p.setPhysicsEngineParameter(deterministicOverlappingPairs=1,numSolverIterations=100,contactBreakingThreshold=.001,physicsClientId=self.cid)
  # All physical collisions are enabled, including robot-small and robot-large.
  for uid in self.names:
   for link in range(-1,p.getNumJoints(uid,physicsClientId=self.cid)):p.setCollisionFilterGroupMask(uid,link,1,1,physicsClientId=self.cid)
  for n,u in self.uid.items():p.changeDynamics(u,-1,mass=.20 if n=='large'else .025,lateralFriction=.8,restitution=0,linearDamping=.04,angularDamping=.04,rollingFriction=.001,spinningFriction=.001,physicsClientId=self.cid)
  ob=ad.observe()['robot_obs'];quat=p.getQuaternionFromEuler(ob[3:6])
  for _ in range(120):ad.step([ob[:3].copy(),quat,1])
  self.root=self.private();self.valid=all(v['upward_support']and np.linalg.norm(v['velocity'])<.005 for v in self.root['objects'].values())
  self.guard=Guard(self);self.robot.mixed_ik.get_bullet_ik=self.guard.runtime
  # Low-gain position servo avoids settling each 30 Hz target as an impulse.
  # Joint motors still integrate through Bullet; no joint/body resets occur.
  def apply_action(action):
   xyz,quat,grip=action;self.robot.gripper_action=int(grip);q=self.guard.runtime(np.asarray(xyz),np.asarray(quat))
   for j,v in zip(self.robot.arm_joint_ids,q):p.setJointMotorControl2(self.robot.robot_uid,j,p.POSITION_CONTROL,targetPosition=float(v),force=100.,maxVelocity=.35,positionGain=.01,velocityGain=1.,physicsClientId=self.cid)
   self.robot.control_gripper(int(grip))
  self.robot.apply_action=apply_action
  if capture:self.record('initial')
 def robot_obs(self):return self.ad.env.get_state_obs()['robot_obs']
 def constraints(self):
  out=[]
  for i in range(p.getNumConstraints(physicsClientId=self.cid)):
   u=p.getConstraintUniqueId(i,physicsClientId=self.cid);c=p.getConstraintInfo(u,physicsClientId=self.cid);out.append({'id':u,'parent':self.names.get(c[0],str(c[0])),'parent_uid':c[0],'parent_link':c[1],'child':self.names.get(c[2],str(c[2])),'child_uid':c[2],'child_link':c[3],'type':c[4]})
  assert not any({q['parent'],q['child']}=={'large','small'}for q in out)
  return out
 def contacts(self):
  cs=[]
  for c in p.getContactPoints(physicsClientId=self.cid):
   if c[1] not in self.names or c[2]not in self.names:continue
   if c[1]==c[2]:continue
   cs.append({'bodyA':self.names[c[1]],'bodyB':self.names[c[2]],'linkA':c[3],'linkB':c[4],'distance':c[8],'force_N':c[9],'normal_on_B':list(c[7]),'pointA':list(c[5]),'pointB':list(c[6])})
  return cs
 def geometry(self):
  pairs=[('robot',n,self.robot.robot_uid,u)for n,u in self.uid.items()]+[('large','small',self.uid['large'],self.uid['small'])]
  for name,u in self.uid.items():
   for obj in self.ad.env.scene.fixed_objects:pairs.append((name,'fixed:'+obj.name,u,obj.uid))
  cs=[]
  for a,b,ua,ub in pairs:
   for c in p.getClosestPoints(ua,ub,0,physicsClientId=self.cid):cs.append({'a':a,'b':b,'linkA':c[3],'linkB':c[4],'penetration_m':max(0.,-c[8]),'pointA':list(c[5]),'pointB':list(c[6])})
  return cs
 def private(self):
  cs=self.contacts();objects={}
  for n,u in self.uid.items():
   pos,orn=p.getBasePositionAndOrientation(u,physicsClientId=self.cid);vel,ang=p.getBaseVelocity(u,physicsClientId=self.cid);up=False
   for c in cs:
    if c['bodyA']==n and c['bodyB']!='robot'and c['force_N']>.02 and c['normal_on_B'][2]>.5:up=True
    if c['bodyB']==n and c['bodyA']!='robot'and c['force_N']>.02 and c['normal_on_B'][2]<-.5:up=True
   objects[n]={'position':list(pos),'orientation':list(orn),'velocity':list(vel),'angular_velocity':list(ang),'region':region(pos),'held':bool(self.held==n and not up),'upward_support':up,'aabb':[list(a)for a in p.getAABB(u,physicsClientId=self.cid)]}
  tcp=p.getLinkState(self.robot.robot_uid,self.robot.tcp_link_id,computeForwardKinematics=True,physicsClientId=self.cid)
  return {'objects':objects,'contacts':cs,'constraints':self.constraints(),'tcp_position':list(tcp[0]),'phase':self.stage}
 def record(self,tag=None):
  ob=self.ad.history[-1];self.frames.append({'rgb_static':ob['rgb_obs']['rgb_static'].copy(),'rgb_gripper':ob['rgb_obs']['rgb_gripper'].copy(),'state':self.private(),'control':len(self.trace),'tag':tag})
 def step(self,xyz,quat,grip):
  if self.steps>=1200:raise RuntimeError('control_budget')
  self.ad.step([np.asarray(xyz),np.asarray(quat),int(grip)]);self.steps+=1;state=self.private();geo=self.geometry();self.trace.append({'command':{'xyz':list(xyz),'grip':grip},'state':state,'geometry':geo})
  if self.capture and len(self.trace)%2==0:self.record()
  if max([0.]+[c['penetration_m']for c in geo])>.002:raise RuntimeError('robot_or_object_penetration_over_2mm')
  # Unintended contacts are a control failure, even with a small penetration.
  active=getattr(self,'picking',None)or self.held
  for c in state['contacts']:
   if c['bodyA']=='robot'and c['bodyB']in self.uid:
    obj=c['bodyB'];link=c['linkA']
   elif c['bodyB']=='robot'and c['bodyA']in self.uid:obj=c['bodyA'];link=c['linkB']
   else:continue
   if c['force_N']>.05 and (obj!=active or link not in self.fingers):raise RuntimeError('unintended_robot_contact:'+obj+':'+str(link))
 def move(self,xyz,grip):
  xyz=np.array(xyz,float);cur=self.robot_obs()[:3];q=p.getQuaternionFromEuler([np.pi,0,0]);n=max(1,int(np.ceil(np.linalg.norm(xyz-cur)/.001)))
  # Minimum-jerk start/stop for every object and arrangement; no carry rule.
  for t in np.linspace(0,1,n+1)[1:]:
   f=10*t**3-15*t**4+6*t**5;self.step(cur+(xyz-cur)*f,q,grip)
  for _ in range(32):
   self.step(xyz,q,grip)
   if np.linalg.norm(self.robot_obs()[:3]-xyz)<.004:break
  if np.linalg.norm(self.robot_obs()[:3]-xyz)>.012:raise RuntimeError('tcp_target_error')
 def hold(self,n,grip):
  ob=self.robot_obs();q=p.getQuaternionFromEuler(ob[3:6])
  for _ in range(n):self.step(ob[:3],q,grip)
 def phase(self,x):self.stage=x;self.phases.append({'phase':x,'control':len(self.trace),'state':self.private()})
 def bilateral(self,n):
  cs=p.getContactPoints(self.robot.robot_uid,self.uid[n],physicsClientId=self.cid);out=[]
  for link in self.fingers:
   q=[c for c in cs if c[3]==link and c[9]>.05];out.append({'link':link,'force_N':sum(c[9]for c in q),'normal':list(np.mean([c[7]for c in q],0))if q else None,'contact_points':[list(c[6])for c in q]})
  ok=all(x['force_N']>.05 for x in out)
  if ok:ok=np.dot(out[0]['normal'],out[1]['normal'])<-.7
  return bool(ok),out
 def pick(self,n):
  if self.held is not None:raise RuntimeError('not_empty')
  before=self.private()['objects'];obj=before[n];pos=np.array(obj['position']);yaw=p.getEulerFromQuaternion(obj['orientation'])[2]
  # Exposed end of large box; small occupies opposite end when stacked.
  offset=np.array([-.030*np.cos(yaw),-.030*np.sin(yaw),.020])if n=='large'else np.array([0.,0.,.008])
  grasp=pos+offset;self.picking=n;self.phase('approach_exposed_side')
  cur=self.robot_obs();ok,audit=self.guard.path([[*cur[:2],.66],[*grasp[:2],.66],grasp],[1,1,1])
  if not ok:raise RuntimeError('grasp_path_unreachable:'+str(audit[-1]))
  self.move([*cur[:2],.66],1);self.move([*grasp[:2],.66],1);self.move(grasp,1);self.phase('closing_on_real_sides');self.hold(65,-1)
  valid,cs=self.bilateral(n);audit={'object':n,'valid':valid,'bilateral_contacts':cs,'state_before_attach':self.private(),'geometry_before_attach':self.geometry()};self.grasp_audits.append(audit)
  if not valid:raise RuntimeError('no_opposing_bilateral_finger_contact')
  if self.assist:
   link=p.getLinkState(self.robot.robot_uid,self.robot.tcp_link_id,physicsClientId=self.cid);body=p.getBasePositionAndOrientation(self.uid[n],physicsClientId=self.cid);inv=p.invertTransform(link[0],link[1]);rel=p.multiplyTransforms(inv[0],inv[1],body[0],body[1]);self.constraint=p.createConstraint(self.robot.robot_uid,self.robot.tcp_link_id,self.uid[n],-1,p.JOINT_FIXED,[0,0,0],rel[0],[0,0,0],parentFrameOrientation=rel[1],childFrameOrientation=[0,0,0,1],physicsClientId=self.cid);p.changeConstraint(self.constraint,maxForce=50,physicsClientId=self.cid)
  self.held=n;self.offset=np.array(self.private()['tcp_position'])-np.array(self.private()['objects'][n]['position']);self.phase('contact_validated_lift');self.move([*grasp[:2],.66],-1);self.hold(20,-1);after=self.private()['objects']
  if after[n]['position'][2]-before[n]['position'][2]<.06 or not after[n]['held']:raise RuntimeError('object_not_held_after_lift')
  if n=='small'and after['large']['position'][2]-before['large']['position'][2]>.003:raise RuntimeError('unintended_large_lift')
  self.picking=None;self.phase('pick_verified')
 def place(self,n,target):
  if self.held!=n:raise RuntimeError('wrong_held_object')
  objs=self.private()['objects'];pos=np.array(objs[n]['position']);zhalf=SIZE[n][2]/2
  if target=='large_top':
   if n!='small':raise RuntimeError('illegal_target')
   b=objs['large'];yaw=p.getEulerFromQuaternion(b['orientation'])[2];xy=np.array(b['position'][:2])+np.array([.030*np.cos(yaw),.030*np.sin(yaw)]);surface=b['aabb'][1][2]
  else:
   reg=0 if target=='A'else 1
   candidates=([[-.08,-.14],[-.16,-.14],[.0,-.14]]if reg==0 else [[.18,-.14],[.24,-.14],[.30,-.14]])if n=='large'else([[-.22,-.14],[.04,-.14],[-.20,-.14],[-.17,-.14]]if reg==0 else[[.32,-.14],[.34,-.14],[.14,-.14],[.30,-.08]])
   other=objs['small'if n=='large'else'large'];xy=None
   for cand in candidates:
    delta=np.abs(np.array(cand)-np.array(other['position'][:2]));threshold=(SIZE[n][:2]+SIZE['small'if n=='large'else'large'][:2])/2+.01
    # During loaded transport the supported small is above the target large;
    # this current-state clearance only screens table objects at destination.
    if not np.any(delta>threshold):continue
    desired_test=np.array([*cand,.44+zhalf+.002]);tcp_test=desired_test+self.offset;current=self.robot_obs();reachable,audit=self.guard.path([[*current[:2],.66],[*tcp_test[:2],.66],tcp_test,tcp_test,[*tcp_test[:2],.66]],[-1,-1,-1,1,1],destination_only_objects=True)
    self.phases.append({'phase':'placement_candidate_audit','target':cand,'valid':reachable,'last':audit[-1]if audit else None,'control':len(self.trace)})
    if reachable:xy=cand;break
   if xy is None:raise RuntimeError('no_clear_table_target')
   surface=.44
  desired=np.array([*xy,surface+zhalf+.002]);tcp=desired+self.offset;self.phase('transfer');cur=self.robot_obs();self.move([*cur[:2],.66],-1);self.move([*tcp[:2],.66],-1);self.move(tcp,-1);self.hold(6,-1);self.phase('release')
  if self.constraint is not None:p.removeConstraint(self.constraint,physicsClientId=self.cid);self.constraint=None
  self.held=None;self.picking=n # Fingers may contact the releasing target until open.
  self.hold(65,1);cur=self.robot_obs();self.move([*cur[:2],.66],1);self.picking=None;self.hold(24,1)
  a=self.private()['objects'][n]
  if not a['upward_support']or a['held']:raise RuntimeError('release_not_supported')
  if target!='large_top'and a['region']!=(0 if target=='A'else 1):raise RuntimeError('wrong_release_region')
  self.phase('place_verified')
 def execute(self,action):
  start=len(self.trace);self.steps=0;status='completed';reason=None
  try:self.pick(action[1])if action[0]=='Pick'else self.place(action[1],action[2])
  except RuntimeError as e:status='failed';reason=str(e)
  self.ad.observe()
  if self.capture:self.record('boundary')
  return {'action':action,'status':status,'reason':reason,'controls':self.steps,'state':self.private(),'trace_range':[start,len(self.trace)]}
 def window(self,path):
  path.mkdir(parents=True,exist_ok=False);out=[]
  for i,ob in enumerate(self.ad.history):
   q=path/f'{i}.npz';np.savez_compressed(q,rgb_static=ob['rgb_obs']['rgb_static'],rgb_gripper=ob['rgb_obs']['rgb_gripper'],robot_obs=ob['robot_obs']);out.append(str(q))
  return out
 def close(self):
  if self.guard:self.guard.close();self.guard=None
  if getattr(self,'ad',None):self.ad.close();self.ad=None
