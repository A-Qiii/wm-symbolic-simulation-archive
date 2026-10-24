"""Simulator-private physical executor. Not a world model or high-level planner.

The only action path is CalvinAdapter.step -> stock robot IK/motors. All GT
queries stay here. This module never resets an object, alters dynamics, attaches
an object, disables collisions or applies external forces.
"""
from pathlib import Path
import math, json
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation, Slerp

class ExecutionFailure(RuntimeError): pass

class PhysicalOracle:
    VERSION='physical_oracle_v1_candidate3'
    def execute(self,skill):
        """Public integration boundary: observations only, never GT diagnostics.

        Existing WM collection remains paused and is not wired to this class.
        Standalone acceptance uses run() only for evaluator-private audit.
        """
        self.last_audit_private=self.run(skill)
        return self.a.observe()
    def __init__(self, adapter, recorder=None, max_controls=900):
        self.a=adapter;self.e=adapter.env;self.cid=self.e.cid;self.robot=self.e.robot
        self.objects={o.name:o for o in self.e.scene.movable_objects}
        self.table=self.e.scene.fixed_objects[0]
        # CALVIN's loaded gripper includes fixed contact pads. A pad's index is
        # not necessarily the index of its actuated finger joint.
        joints={i:p.getJointInfo(self.robot.robot_uid,i,physicsClientId=self.cid) for i in range(p.getNumJoints(self.robot.robot_uid,physicsClientId=self.cid))}
        self.finger_families=[]
        for finger in self.robot.gripper_joint_ids:
            family={finger}
            for _ in joints:
                family.update(i for i,q in joints.items() if q[16] in family)
            self.finger_families.append(family)
        self.links={p.getJointInfo(self.table.uid,i,physicsClientId=self.cid)[12].decode():i for i in range(p.getNumJoints(self.table.uid,physicsClientId=self.cid))}
        self.recorder=recorder;self.max_controls=max_controls;self.steps=0;self.events=[];self.phase='idle'

    def pose(self):return self.a.observe()['robot_obs'][:6]
    def xyz(self,name):return np.asarray(p.getBasePositionAndOrientation(self.objects[name].uid,physicsClientId=self.cid)[0])
    def joint(self,link):return float(p.getJointState(self.table.uid,self.links[link],physicsClientId=self.cid)[0])
    def contacts(self,name):return p.getContactPoints(bodyA=self.objects[name].uid,physicsClientId=self.cid)
    def held(self):
        result=[]
        for name,obj in self.objects.items():
            contacts=[c for c in self.contacts(name) if c[9]>.01]
            robot=[c for c in contacts if c[2]==self.robot.robot_uid]
            fingerlinks={c[4] for c in robot}
            if all(fingerlinks&family for family in self.finger_families) and not any(c[2]!=self.robot.robot_uid for c in contacts):result.append(name)
        return result
    def on_table(self,name):return any(c[2]==self.table.uid and c[4]==-1 and c[9]>.01 for c in self.contacts(name))
    def in_drawer(self,name):return any(c[2]==self.table.uid and c[4]==self.links['drawer_link'] and c[9]>.01 for c in self.contacts(name))
    def step(self,xyz,quat,grip):
        if self.steps>=self.max_controls:raise ExecutionFailure('control_budget_exhausted')
        obs=self.a.step([np.asarray(xyz),np.asarray(quat),int(grip)]);self.steps+=1
        if self.recorder:self.recorder(obs,{'step':self.steps,'phase':self.phase,'target_xyz':list(xyz),'target_quaternion':list(quat),'gripper':int(grip)})
        return obs
    def move(self,xyz,orn,grip,phase,speed=.004,settle=12):
        self.phase=phase;start=self.pose();xyz=np.asarray(xyz,float)
        qtarget=Rotation.from_euler('xyz',orn) if len(orn)==3 else Rotation.from_quat(orn)
        qstart=Rotation.from_euler('xyz',start[3:]);angle=(qtarget*qstart.inv()).magnitude()
        n=max(1,int(np.ceil(np.linalg.norm(xyz-start[:3])/speed)),int(np.ceil(angle/.06)))
        curve=Slerp([0,1],Rotation.from_quat([qstart.as_quat(),qtarget.as_quat()]))
        for t in np.linspace(0,1,n+1)[1:]:self.step(start[:3]+t*(xyz-start[:3]),curve(t).as_quat(),grip)
        for _ in range(settle):self.step(xyz,qtarget.as_quat(),grip)
        error=float(np.linalg.norm(self.pose()[:3]-xyz));self.events.append({'phase':phase,'end_step':self.steps,'tcp_error_m':error})
        if error>.045:raise ExecutionFailure(f'unreachable_or_obstructed:{phase}:{error:.4f}')
    def hold(self,n,grip):
        cur=self.pose();q=Rotation.from_euler('xyz',cur[3:]).as_quat()
        for _ in range(n):self.step(cur[:3],q,grip)
    def safe_height(self,orn,grip):
        cur=self.pose();self.move([cur[0],cur[1],max(cur[2],.62)],orn,grip,'raise_clearance')
    def block_orn(self,name):
        orn=p.getBasePositionAndOrientation(self.objects[name].uid,physicsClientId=self.cid)[1]
        yaw=p.getEulerFromQuaternion(orn)[2]
        # Closing axis is local Y; align to the block's narrow dimension.
        current=self.pose()[5];yaw=min([yaw+k*np.pi for k in range(-3,4)],key=lambda v:abs(v-current))
        return [np.pi,0,yaw]
    def drawer_handle(self):
        # Handle mesh dimensions are identical in local coordinates across A/B/C.
        state=p.getLinkState(self.table.uid,self.links['drawer_link'],physicsClientId=self.cid)
        pos,_=p.multiplyTransforms(state[4],state[5],[0,.05*.8,.315*.8],[0,0,0,1])
        return np.asarray(pos)
    def drawer(self,opening):
        if not opening:
            # Causal development correction: a downward grasp hits the table
            # lip near closed, and its rear finger hooks the handle on retreat.
            # Push the front of the handle with a horizontally oriented closed
            # gripper; the palm remains in front of the table throughout.
            orn=[-np.pi/2,0,0];h=self.drawer_handle();self.safe_height(orn,-1)
            self.move([h[0],h[1]-.09,.58],orn,-1,'in_front_of_handle')
            self.move(h+[0,-.09,0],orn,-1,'lower_in_front_of_handle',speed=.003)
            end=h+[0,self.joint('drawer_link')+.006,0]
            self.move(end,orn,-1,'horizontal_push_drawer',speed=.0025,settle=20)
            cur=self.pose();self.move(cur[:3]+[0,-.10,0],orn,-1,'withdraw_front_of_handle')
            self.safe_height(orn,1)
            return
        orn=[np.pi,0,0];h=self.drawer_handle();self.safe_height(orn,1)
        self.move([h[0],h[1],.58],orn,1,'above_handle')
        self.move(h,orn,1,'descend_to_handle',speed=.003)
        self.hold(18,-1)
        start=self.joint('drawer_link');target=.205 if opening else .005
        end=h+np.array([0,-(target-start),0])
        self.move(end,orn,-1,'pull_drawer' if opening else 'push_drawer',speed=.0025,settle=20)
        self.hold(12,1)
        # First move away from the front, then lift to avoid the tabletop lip.
        cur=self.pose();self.move(cur[:3]+[0,-.055,0],orn,1,'clear_handle')
        self.safe_height(orn,1)
    def lift(self,name,attempt=0):
        orn=self.block_orn(name)
        if attempt:orn[2]+=np.pi
        self.safe_height(orn,1);pos=self.xyz(name)
        self.move([pos[0],pos[1],.59],orn,1,'above_named_block')
        self.move(pos+[0,0,.003],orn,1,'descend_named_block',speed=.0025)
        self.hold(20,-1)
        self.move(pos+[0,0,.14],orn,-1,'lift_named_block',speed=.0025,settle=18)
    def place(self,name):
        orn=self.pose()[3:];self.safe_height(orn,-1)
        state=p.getLinkState(self.table.uid,self.links['drawer_link'],physicsClientId=self.cid)
        center,_=p.multiplyTransforms(state[4],state[5],[0,.045*.8,.10*.8],[0,0,0,1])
        center=np.asarray(center)
        # Open drawer interior between the front face and tabletop edge.
        self.move([center[0],center[1],.59],orn,-1,'above_drawer_interior')
        self.move([center[0],center[1],.405],orn,-1,'lower_into_drawer',speed=.0025)
        self.hold(24,1)
        cur=self.pose();self.move([cur[0],cur[1],.61],orn,1,'withdraw_after_release',speed=.003)
        self.hold(12,1)
    def run(self,skill):
        self.steps=0;self.events=[];attempts=0;reason=None;start={k:self.xyz(k).copy() for k in self.objects}
        if skill not in ['open_drawer','close_drawer','place_in_drawer'] and not (skill.startswith('lift_') and skill.endswith('_block_table')):
            return {'status':'refused','reason':'unsupported_skill','controls':0,'attempts':0,'retries':0}
        held=self.held();target=None
        if skill=='place_in_drawer':
            if len(held)!=1:reason='requires_exactly_one_held_block'
            elif self.joint('drawer_link')<.15:reason='requires_open_drawer'
            else:target=held[0]
        elif skill in ['open_drawer','close_drawer']:
            if held:reason='requires_empty_gripper'
            elif skill=='open_drawer' and self.joint('drawer_link')>.15:reason='drawer_already_open'
            elif skill=='close_drawer' and self.joint('drawer_link')<.02:reason='drawer_already_closed'
        else:
            target='block_'+skill.split('_')[1]
            if target not in self.objects:reason='unknown_named_object'
            elif held:reason='requires_empty_gripper'
            elif not self.on_table(target):reason='requires_named_block_on_table'
        if reason:return {'status':'refused','reason':reason,'controls':0,'attempts':0,'retries':0,'target':target}
        success=False
        try:
            for attempt in range(2):
                attempts=attempt+1
                if skill in ['open_drawer','close_drawer']:
                    self.drawer(skill=='open_drawer')
                    success=self.joint('drawer_link')>.18 if skill=='open_drawer' else self.joint('drawer_link')<.02
                elif skill=='place_in_drawer':
                    self.place(target);success=self.in_drawer(target) and not self.held()
                else:
                    try:self.lift(target,attempt)
                    except ExecutionFailure as ex:
                        # One bounded geometric alternative: the rectangular
                        # block has the same grasp axis after 180-degree yaw.
                        # Only retry a pre-contact approach failure; never
                        # conceal a dropped object or move prerequisites.
                        if attempt==0 and self.phase in ['above_named_block','raise_clearance'] and self.on_table(target) and not self.held():
                            self.events.append({'phase':'bounded_symmetric_grasp_retry','end_step':self.steps,'cause':str(ex)})
                            continue
                        raise
                    success=target in self.held() and self.xyz(target)[2]-start[target][2]>.07
                if success:
                    # Goal must survive 12 extra physical controls, without
                    # feeding this private completion bit to a world model.
                    self.phase='verify_stable_completion'
                    for _ in range(12):
                        self.hold(1,-1 if skill.startswith('lift_') else 1)
                        if skill=='open_drawer':valid=self.joint('drawer_link')>.18
                        elif skill=='close_drawer':valid=self.joint('drawer_link')<.02
                        elif skill=='place_in_drawer':valid=self.in_drawer(target) and not self.held()
                        else:valid=target in self.held() and self.xyz(target)[2]-start[target][2]>.07
                        if not valid:success=False;reason='unstable_completion';break
                    if success:break
                if skill=='place_in_drawer' or (skill.startswith('lift') and not self.on_table(target)):
                    reason='failed_goal_and_retry_precondition_absent';break
                reason='completion_check_failed'
        except ExecutionFailure as ex:success=False;reason=str(ex)
        return {'status':'success' if success else 'failed','reason':None if success else reason,'controls':self.steps,'attempts':attempts,'retries':max(0,attempts-1),'target':target,'events':self.events,'drawer_position':self.joint('drawer_link'),'held_private':self.held()}
