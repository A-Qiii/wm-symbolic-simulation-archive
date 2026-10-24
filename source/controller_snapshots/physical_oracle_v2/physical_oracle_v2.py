"""GT-private full-geometry grasp planning; execution still uses stock motors.

KinematicScene is a separate, non-executing PyBullet DIRECT client. Reset calls
only position its query geometry; they never touch the execution client.
"""
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation,Slerp
from wm_symbolic.physical_oracle import PhysicalOracle,ExecutionFailure

class KinematicScene:
    def __init__(self,env):
        self.env=env;self.actual=env.cid;self.cid=p.connect(p.DIRECT);r=env.robot;self.r=r
        self.robot=p.loadURDF(r.filename,r.base_position,r.base_orientation,useFixedBase=True,physicsClientId=self.cid)
        self.body_map={};self.name_map={}
        for o in list(env.scene.fixed_objects)+list(env.scene.movable_objects):
            pos,orn=p.getBasePositionAndOrientation(o.uid,physicsClientId=self.actual)
            uid=p.loadURDF(str(o.file),pos,orn,globalScaling=o.global_scaling,useFixedBase=True,physicsClientId=self.cid)
            self.body_map[o.uid]=uid;self.name_map[o.name]=uid
        self.sync()
    def close(self):
        if self.cid is not None:p.disconnect(self.cid);self.cid=None
    def sync(self):
        for j in range(p.getNumJoints(self.r.robot_uid,physicsClientId=self.actual)):
            p.resetJointState(self.robot,j,p.getJointState(self.r.robot_uid,j,physicsClientId=self.actual)[0],physicsClientId=self.cid)
        for actual,ghost in self.body_map.items():
            pose=p.getBasePositionAndOrientation(actual,physicsClientId=self.actual)
            p.resetBasePositionAndOrientation(ghost,*pose,physicsClientId=self.cid)
            for j in range(p.getNumJoints(actual,physicsClientId=self.actual)):
                p.resetJointState(ghost,j,p.getJointState(actual,j,physicsClientId=self.actual)[0],physicsClientId=self.cid)
    def tcp(self):
        state=p.getLinkState(self.robot,self.r.tcp_link_id,computeForwardKinematics=True,physicsClientId=self.cid)
        return np.array(state[0]),Rotation.from_quat(state[1])
    def set_ik(self,pos,quat):
        q=p.calculateInverseKinematics(self.robot,self.r.tcp_link_id,pos,quat,self.r.ll,self.r.ul,self.r.jr,self.r.rp,maxNumIterations=100,residualThreshold=.0001,physicsClientId=self.cid)
        q=np.clip(q[:7],self.r.ll_real,self.r.ul_real)
        for j,v in zip(self.r.arm_joint_ids,q):p.resetJointState(self.robot,j,float(v),physicsClientId=self.cid)
        actual,_=self.tcp();return float(np.linalg.norm(actual-pos))
    def check(self,waypoints,target):
        self.sync();margin_min=float('inf');samples=0;worst_error=0
        for stage,(goal,quat) in enumerate(waypoints):
            start,orn=self.tcp();end=Rotation.from_quat(quat);angle=(end*orn.inv()).magnitude()
            n=max(1,int(np.ceil(np.linalg.norm(np.array(goal)-start)/.006)),int(np.ceil(angle/.07)))
            curve=Slerp([0,1],Rotation.from_quat([orn.as_quat(),end.as_quat()]))
            for t in np.linspace(0,1,n+1)[1:]:
                pos=start+t*(np.array(goal)-start);q=curve(t).as_quat();error=self.set_ik(pos,q);samples+=1;worst_error=max(worst_error,error)
                if error>.018:return {'valid':False,'reason':'IK_target_error','error_m':error,'stage':stage,'samples':samples}
                for name,body in self.name_map.items():
                    if name==target:continue
                    for c in p.getClosestPoints(self.robot,body,distance=.018,physicsClientId=self.cid):
                        if c[3]<0:continue
                        # Keep the entire arm/hand/fingers clear of moving
                        # devices, with a margin for stock IK tracking error.
                        margin=.012 if name=='table' and c[4] in [0,1,2] else .002
                        clearance=c[8]-margin;margin_min=min(margin_min,clearance)
                        if clearance<0:return {'valid':False,'reason':'collision_clearance','obstacle':name,'robot_link':c[3],'obstacle_link':c[4],'distance_m':c[8],'required_margin_m':margin,'stage':stage,'samples':samples}
        return {'valid':True,'samples':samples,'minimum_excess_clearance_m':margin_min if np.isfinite(margin_min) else .018,'max_IK_error_m':worst_error}

class PhysicalOracleV2(PhysicalOracle):
    VERSION='physical_oracle_v2_candidate0'
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.shadow=KinematicScene(self.e);self.planning_audit=[];self.monitor=None;self._plans=None
    def close(self):self.shadow.close()
    def step(self,*args,**kwargs):
        obs=super().step(*args,**kwargs)
        if self.monitor:
            state=self.e.scene.get_obs()
            if np.max(np.abs(state[[0,2,3,4,5]]-self.monitor['scene'][[0,2,3,4,5]]))>.003:
                raise ExecutionFailure('unrequested_device_change')
            for name,old in self.monitor['objects'].items():
                if name not in self.monitor['allowed'] and np.linalg.norm(self.xyz(name)-old)>.01:
                    raise ExecutionFailure('unrequested_object_change:'+name)
        return obs
    def run(self,skill):
        allowed=set()
        if skill.startswith('lift_'):allowed.add('block_'+skill.split('_')[1])
        elif skill=='place_in_drawer':allowed.update(self.held())
        elif skill=='close_drawer':allowed.update(n for n in self.objects if self.in_drawer(n))
        self.monitor={'scene':self.e.scene.get_obs().copy(),'objects':{n:self.xyz(n).copy() for n in self.objects},'allowed':allowed}
        self.planning_audit=[];self._plans=None
        result=super().run(skill);result['planning_private']=self.planning_audit;result['execution_world_reset_count']=0;result['query_world_separate_client']=self.shadow.cid!=self.cid
        self.monitor=None;return result
    def plan_lift(self,name):
        pos=self.xyz(name);baseyaw=self.block_orn(name)[2];cur=self.pose();valid=[];self.phase='plan_full_gripper_clearance'
        # All candidates are fixed before v2 development. Search is private,
        # bounded and contains no physical trial or object manipulation.
        for tilt in [0,15,25,35]:
          for offset in [.003,.010]:
            for flip in [0,1]:
                orn=(Rotation.from_euler('x',tilt,degrees=True)*Rotation.from_euler('xyz',[np.pi,0,baseyaw+flip*np.pi])).as_quat()
                waypoints=[([cur[0],cur[1],max(.67,cur[2])],orn),([pos[0],pos[1],.64],orn),(pos+[0,0,offset],orn),(pos+[0,0,.14],orn)]
                check=self.shadow.check(waypoints,name);row={'tilt_world_x_degrees':tilt,'grasp_z_offset_m':offset,'yaw_flip':flip,**check};self.planning_audit.append(row)
                if check['valid']:valid.append((tilt+.1*flip+offset,waypoints,row))
        valid.sort(key=lambda x:x[0]);self._plans=valid
        if not valid:raise ExecutionFailure('no_collision_free_grasp_candidate')
        return valid
    def lift(self,name,attempt=0):
        if self._plans is None:self.plan_lift(name)
        if attempt>=len(self._plans):raise ExecutionFailure('bounded_grasp_candidates_exhausted')
        _,waypoints,meta=self._plans[attempt];self.events.append({'phase':'selected_clearance_checked_grasp','end_step':self.steps,'candidate':meta})
        (raise_pos,orn),(above,_),(grasp,_),(lift,_)=waypoints
        self.move(raise_pos,orn,1,'raise_clearance')
        self.move(above,orn,1,'above_named_block')
        self.move(grasp,orn,1,'descend_named_block',speed=.0025)
        self.hold(20,-1)
        self.move(lift,orn,-1,'lift_named_block',speed=.0025,settle=18)
