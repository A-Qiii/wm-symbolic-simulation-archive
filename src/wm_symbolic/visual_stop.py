"""Frozen candidate stop/refusal layer. HULC itself and its actions are unchanged."""
import numpy as np
from wm_symbolic.public_geometry import public_frame,visible_colours,drawer_front

class VisualStopper:
 def __init__(self,skill,initial,config):
  self.skill=skill;self.config=config;self.initial_colours=visible_colours(public_frame(initial));self.previous_reason=None;self.streak=0
 def check(self,obs,step):
  c=self.config
  if step<c['minimum_controls'] or step%c['check_every_controls']:return None
  frame=public_frame(obs);reason=None;evidence={'control_step':step,'input_source':'public RGB-D/masks/calibration/robot only','skill':self.skill}
  if self.skill in ['open_drawer','close_drawer']:
   part=drawer_front(frame);travel=None if part is None else c['drawer_closed_front_y_m']-part['front_y_m'];evidence.update(visible_drawer=part,estimated_drawer_travel_m=travel)
   if travel is not None:
    if self.skill=='open_drawer' and travel>=c['drawer_open_min_m']:reason='visual_completed'
    if self.skill=='close_drawer' and -.015<=travel<=c['drawer_closed_max_m']:reason='visual_completed'
  elif self.skill.startswith('lift_'):
   target=next(color for color in ['red','blue','pink'] if '_'+color+'_' in self.skill);current=visible_colours(frame);robot=np.asarray(frame['robot_obs']);evidence.update(target_color=target,visible_colors=current,gripper_width_m=float(robot[6]),gripper_command=float(robot[-1]));held=[];details={}
   for color,value in current.items():
    if color not in self.initial_colours:continue
    point=np.asarray(value['point']);rise=float(point[2]-self.initial_colours[color]['point'][2]);distance=float(np.linalg.norm(point-robot[:3]));details[color]={'visible_rise_m':rise,'distance_to_public_TCP_m':distance}
    if rise>=c['lift_min_rise_m'] and distance<=c['lift_max_TCP_distance_m'] and c['gripper_width_min_m']<=robot[6]<=c['gripper_width_max_m'] and robot[-1]<0:held.append(color)
   evidence.update(holding_candidates=held,holding_geometry=details)
   if held==[target]:reason='visual_completed'
   elif len(held)==1 and held[0]!=target:reason='wrong_visible_object_refusal'
   elif len(held)>1:reason='ambiguous_visible_holding_refusal'
  else:evidence['unsupported_completion_detector']=True
  self.streak=self.streak+1 if reason is not None and reason==self.previous_reason else (1 if reason else 0);self.previous_reason=reason;evidence.update(candidate_reason=reason,consecutive_checks=self.streak,decision=reason if self.streak>=c['consecutive_checks'] else None)
  return evidence
