"""Add bounded, fully checked alternate approach routes to the frozen v2 core."""
import numpy as np
from scipy.spatial.transform import Rotation
from wm_symbolic.physical_oracle import ExecutionFailure
from wm_symbolic.physical_oracle_v2 import PhysicalOracleV2

class PhysicalOracleV3(PhysicalOracleV2):
    VERSION='physical_oracle_v3_candidate0'
    def plan_lift(self,name):
        try:return super().plan_lift(name)
        except ExecutionFailure as error:
            if str(error)!='no_collision_free_grasp_candidate':raise
        pos=self.xyz(name);baseyaw=self.block_orn(name)[2];cur=self.pose();valid=[]
        front=[float(self.drawer_handle()[0]),-.50,.62]
        prefix=[([cur[0],cur[1],max(.67,cur[2])],Rotation.from_euler('xyz',cur[3:]).as_quat()),(front,Rotation.from_euler('xyz',[np.pi,0,0]).as_quat())]
        for tilt in [0,15,25,35]:
          for offset in [.003,.010]:
            for flip in [0,1]:
                orn=(Rotation.from_euler('x',tilt,degrees=True)*Rotation.from_euler('xyz',[np.pi,0,baseyaw+flip*np.pi])).as_quat()
                waypoints=prefix+[([front[0],front[1],.67],orn),([pos[0],pos[1],.64],orn),(pos+[0,0,offset],orn),(pos+[0,0,.14],orn)]
                check=self.shadow.check(waypoints,name)
                row={'route':'front_clearance','tilt_world_x_degrees':tilt,'grasp_z_offset_m':offset,'yaw_flip':flip,**check};self.planning_audit.append(row)
                if check['valid']:valid.append((tilt+.1*flip+offset,waypoints,row))
        valid.sort(key=lambda x:x[0]);self._plans=valid
        if not valid:raise ExecutionFailure('no_collision_free_grasp_or_front_route_candidate')
        return valid
    def lift(self,name,attempt=0):
        if self._plans is None:self.plan_lift(name)
        if attempt>=len(self._plans):raise ExecutionFailure('bounded_grasp_candidates_exhausted')
        _,waypoints,meta=self._plans[attempt]
        if len(waypoints)==4:return super().lift(name,attempt)
        self.events.append({'phase':'selected_clearance_checked_grasp','end_step':self.steps,'candidate':meta})
        for i,(pos,orn) in enumerate(waypoints[:-2]):
            phase='raise_clearance' if i==0 else 'above_named_block' if i==len(waypoints)-3 else 'front_clearance_route'
            self.move(pos,orn,1,phase)
        (grasp,orn),(lift,_)=waypoints[-2:]
        self.move(grasp,orn,1,'descend_named_block',speed=.0025);self.hold(20,-1)
        self.move(lift,orn,-1,'lift_named_block',speed=.0025,settle=18)
