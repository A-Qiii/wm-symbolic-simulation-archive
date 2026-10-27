"""Private control-layer shadow kinematics,never a world-model successor.
Separate DIRECT client;query robot/furniture collision and named target paths.
No real-world body pose is reset or teleported.
"""
from paths import CALVIN_ENV
from paths import ROOT as B, RUNTIME as P
from pathlib import Path
import numpy as np,pybullet as p
class Guard:
 def __init__(self,scene):
  self.s=scene;self.cid=p.connect(p.DIRECT);r=scene.robot;file=CALVIN_ENV/'data'/r.filename;self.robot=p.loadURDF(str(file),r.base_position,r.base_orientation,useFixedBase=True,physicsClientId=self.cid);self.fixed=[]
  for o in scene.ad.env.scene.fixed_objects:
   u=p.loadURDF(str(o.file),o.initial_pos,o.initial_orn,globalScaling=o.global_scaling,physicsClientId=self.cid);self.fixed.append((u,o.name))
   for j in range(p.getNumJoints(u,physicsClientId=self.cid)):p.resetJointState(u,j,p.getJointState(o.uid,j,physicsClientId=scene.cid)[0],physicsClientId=self.cid)
  self.arm=list(r.arm_joint_ids);self.grip=list(r.gripper_joint_ids);self.dofs=self.arm+self.grip;self.lo=list(r.ll_real)+[r.gripper_joint_limits[0]]*2;self.hi=list(r.ul_real)+[r.gripper_joint_limits[1]]*2;self.ranges=(np.array(self.hi)-np.array(self.lo)).tolist();self.last_diagnostic=None
 def seed(self):return [p.getJointState(self.s.robot.robot_uid,j,physicsClientId=self.s.cid)[0]for j in self.dofs]
 def setq(self,q):
  for j,v in zip(self.dofs,q):p.resetJointState(self.robot,j,float(v),physicsClientId=self.cid)
 def solve(self,target,quat,seed=None,grip=None,collision=False):
  q=self.seed()if seed is None else list(seed)
  if grip is not None:q[-2:]=[self.s.robot.gripper_joint_limits[1 if grip==1 else 0]]*2
  self.setq(q)
  for _ in range(3):
   result=p.calculateInverseKinematics(self.robot,self.s.robot.tcp_link_id,target,quat,lowerLimits=self.lo,upperLimits=self.hi,jointRanges=self.ranges,restPoses=q,maxNumIterations=150,residualThreshold=1e-5,physicsClientId=self.cid);q=np.clip(np.array(result[:9]),self.lo,self.hi).tolist();q[-2:]=[self.s.robot.gripper_joint_limits[1 if grip==1 else 0]]*2 if grip is not None else q[-2:];self.setq(q);pose=p.getLinkState(self.robot,self.s.robot.tcp_link_id,computeForwardKinematics=True,physicsClientId=self.cid);error=float(np.linalg.norm(np.array(pose[0])-target));angle=float(2*np.arccos(min(1.,abs(np.dot(pose[1],quat)))))
   if error<.003 and angle<.03:break
  contacts=[]
  if collision:
   p.performCollisionDetection(physicsClientId=self.cid)
   for uid,name in self.fixed:
    for c in p.getClosestPoints(self.robot,uid,.005,physicsClientId=self.cid):
     if c[3]>=0 and c[8]<.001:contacts.append({'robot_link':c[3],'fixed':name,'fixed_link':c[4],'distance':c[8]})
  ok=error<=.006 and angle<=.06 and not contacts and all(a-1e-6<=x<=b+1e-6 for x,a,b in zip(q,self.lo,self.hi));diag={'valid':ok,'position_error_m':error,'orientation_error_rad':angle,'collisions':contacts,'target':list(target),'joint_violations':[{'joint':j,'q':x,'lower':a,'upper':b}for j,(x,a,b)in enumerate(zip(q,self.lo,self.hi))if not a-1e-6<=x<=b+1e-6]};self.last_diagnostic=diag;return q,diag
 def path(self,waypoints,grips,destination_only_objects=False):
  # Include the current non-target object as an obstacle in the shadow client.
  # All resets below affect only this separate kinematic world.
  saved_fixed=self.fixed.copy();temporary=[]
  if destination_only_objects:
   ok,details=self._path(waypoints,grips)
   if not ok:return ok,details
  active=getattr(self.s,'picking',None)or self.s.held
  for name,uid in self.s.uid.items():
   if name==active:continue
   pos,orn=p.getBasePositionAndOrientation(uid,physicsClientId=self.s.cid)
   clone=p.loadURDF(str((B/'assets')/f'{name}.urdf'),pos,orn,useFixedBase=True,globalScaling=.8,physicsClientId=self.cid);temporary.append(clone);self.fixed.append((clone,'movable:'+name))
  try:
   if not destination_only_objects:return self._path(waypoints,grips)
   # Non-target objects can move under real contact while carrying another
   # object. Query destination poses rather than freezing them along a path.
   # Actual runtime monitors every robot/object pair throughout transfer.
   seed=self.seed();details=[];quat=p.getQuaternionFromEuler([np.pi,0,0])
   for target,grip in list(zip(waypoints,grips))[1:]:
    seed,d=self.solve(np.array(target),quat,seed,grip,True);details.append(d)
    if not d['valid']:return False,details
   return True,details
  finally:
   self.fixed=saved_fixed
   for uid in temporary:p.removeBody(uid,physicsClientId=self.cid)
 def _path(self,waypoints,grips):
  seed=self.seed();start=np.array(self.s.robot_obs()[:3]);quat=p.getQuaternionFromEuler([np.pi,0,0]);details=[]
  for target,grip in zip(waypoints,grips):
   target=np.array(target);n=max(1,int(np.ceil(np.linalg.norm(target-start)/.02)))
   for f in np.linspace(0,1,n+1)[1:]:
    seed,d=self.solve(start+(target-start)*f,quat,seed,grip,True)
    if not d['valid']:return False,details+[d]
   details.append(d);start=target
  return True,details
 def runtime(self,target,quat):
  q,diag=self.solve(target,quat,collision=False)
  if not diag['valid']:raise RuntimeError('ik_pose_unreachable:'+str(diag))
  return np.array(q[:7])
 def close(self):
  if p.isConnected(self.cid):p.disconnect(self.cid)
