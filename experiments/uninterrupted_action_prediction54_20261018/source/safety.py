"""Motor-only execution guard and prospective nine-width shadow query."""
from contextlib import contextmanager
import numpy as np
DENIED=['resetBasePositionAndOrientation','resetBaseVelocity','resetJointState','changeDynamics','setCollisionFilterPair','setCollisionFilterGroupMask','createConstraint','removeConstraint','changeConstraint','applyExternalForce','applyExternalTorque','setGravity','setPhysicsEngineParameter','resetSimulation','restoreState','loadBullet','saveState','saveBullet','loadURDF','loadSDF','loadMJCF','removeBody','setRealTimeSimulation']
@contextmanager
def physical_guard(pb,env,shadow,audit):
 originals={n:getattr(pb,n)for n in DENIED+['setJointMotorControl2','setJointMotorControlArray']}
 def denied(*a,_name='',**kw):
  audit['forbidden_attempts'].append({'call':_name,'client':kw.get('physicsClientId',0)})
  raise RuntimeError('Forbidden execution-world mutation: '+_name)
 def query(*a,_name='',**kw):
  if kw.get('physicsClientId')!=shadow.cid or shadow.cid==env.cid:return denied(*a,_name=_name,**kw)
  audit['shadow_only_reset_calls']+=1;return originals[_name](*a,**kw)
 def motor(*a,_name='',**kw):
  body=kw.get('bodyIndex',a[0]if a else None)
  if body!=env.robot.robot_uid or kw.get('physicsClientId',0)!=env.cid:return denied(*a,_name=_name,**kw)
  audit['native_robot_motor_calls']+=1;return originals[_name](*a,**kw)
 try:
  for n in DENIED:setattr(pb,n,lambda *a,_name=n,**kw:denied(*a,_name=_name,**kw))
  for n in ['resetJointState','resetBasePositionAndOrientation']:setattr(pb,n,lambda *a,_name=n,**kw:query(*a,_name=_name,**kw))
  for n in ['setJointMotorControl2','setJointMotorControlArray']:setattr(pb,n,lambda *a,_name=n,**kw:motor(*a,_name=_name,**kw))
  yield
 finally:
  for n,v in originals.items():setattr(pb,n,v)
def opening_sweep(pb,shadow,target):
 assert shadow.cid!=shadow.actual
 shadow.sync();r=shadow.r
 start=np.array([pb.getJointState(r.robot_uid,j,physicsClientId=shadow.actual)[0]for j in r.gripper_joint_ids]);end=np.full(len(start),r.gripper_joint_limits[1]);rows=[];minimum=float('inf')
 for k,t in enumerate(np.linspace(0,1,9)):
  q=start+t*(end-start)
  for j,v in zip(r.gripper_joint_ids,q):pb.resetJointState(shadow.robot,j,float(v),physicsClientId=shadow.cid)
  for name,body in shadow.name_map.items():
   if name==target:continue
   for c in pb.getClosestPoints(shadow.robot,body,distance=.018,physicsClientId=shadow.cid):
    if c[3]<0:continue
    margin=.012 if name=='table'and c[4]in[0,1,2]else.002;minimum=min(minimum,float(c[8]-margin))
    if c[8]<margin:return {'valid':False,'reason':'open_gripper_sweep_clearance','sample':k,'finger_positions':q.tolist(),'obstacle':name,'robot_link':c[3],'obstacle_link':c[4],'distance_m':c[8],'required_margin_m':margin,'samples_tested':k+1,'query_client_separate':True}
  rows.append({'fraction':float(t),'finger_positions':q.tolist()})
 return {'valid':True,'samples_tested':9,'samples':rows,'minimum_excess_clearance_m':minimum if np.isfinite(minimum)else.018,'query_client_separate':True,'limitation':'prespecified discrete sweep, not a continuous collision proof'}
