"""Every physics substep, including robot/object geometry omitted by old checks."""
import numpy as np,pybullet as p
from block_env import region
class Monitor:
 def __init__(self,scene,keep_samples=True):
  self.scene=scene;self.original=p.stepSimulation;self.samples=[];self.keep_samples=keep_samples;self.n=0;self.peak=0.;self.unintended=[];self.region_changes=[];self.previous_regions=None;self.forbidden_state_seen=False
  def step(*a,**kw):
   r=self.original(*a,**kw)
   if kw.get('physicsClientId',0)!=scene.cid:return r
   geometry=scene.geometry();peak=max([0.]+[g['penetration_m']for g in geometry]);self.peak=max(self.peak,peak);positions={n:list(p.getBasePositionAndOrientation(u,physicsClientId=scene.cid)[0])for n,u in scene.uid.items()};cs=scene.contacts();active=getattr(scene,'picking',None)or scene.held;bad=[]
   for c in cs:
    if c['bodyA']=='robot'and c['bodyB']in scene.uid:obj,link=c['bodyB'],c['linkA']
    elif c['bodyB']=='robot'and c['bodyA']in scene.uid:obj,link=c['bodyA'],c['linkB']
    else:continue
    if c['force_N']>.05 and (obj!=active or link not in scene.fingers):bad.append(c)
   regions={n:region(x)for n,x in positions.items()};up=False
   for c in cs:
    if c['bodyA']=='small'and c['bodyB']!='robot'and c['force_N']>.02 and c['normal_on_B'][2]>.5:up=True
    if c['bodyB']=='small'and c['bodyA']!='robot'and c['force_N']>.02 and c['normal_on_B'][2]<-.5:up=True
   if regions['large']==1 and scene.held=='small'and not up:self.forbidden_state_seen=True
   if regions!=self.previous_regions:self.region_changes.append({'step':self.n,'control':len(scene.trace),'regions':regions});self.previous_regions=regions
   if bad:self.unintended.append({'step':self.n,'phase':scene.stage,'contacts':bad})
   if self.keep_samples:self.samples.append({'step':self.n,'control':len(scene.trace),'phase':scene.stage,'peak_penetration_m':peak,'positions':positions,'regions':regions})
   self.n+=1
   if peak>.002:raise RuntimeError('physics_substep_penetration_over_2mm')
   if bad:raise RuntimeError('physics_substep_unintended_robot_contact:'+str(bad[0]))
   return r
  p.stepSimulation=step
 def close(self):p.stepSimulation=self.original
