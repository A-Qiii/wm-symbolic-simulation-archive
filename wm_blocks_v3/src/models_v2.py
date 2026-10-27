"""One intervention: actor motion uses unified learned object states, not h.
Perception, Held update, interaction form and displacement output are unchanged.
"""
import torch
from torch import nn
from models import Dynamics,RelativeTransition

class StateActorMotion(RelativeTransition):
 def __init__(self):
  super().__init__('motion_matrix')
  # Shared semantic inputs/common layers retain exactly the v1 initialization.
  # New relative/other-size columns begin at zero; no hand-written motion rule.
  old=self.actor[0];new=nn.Linear(30,128)
  with torch.no_grad():
   new.weight.zero_();new.weight[:,:6].copy_(old.weight[:,:6]);new.weight[:,12:].copy_(old.weight[:,262:]);new.bias.copy_(old.bias)
  self.actor[0]=new
 def forward(self,h,p,shape,z,verb,obj,target,scale):
  ix=torch.arange(len(p),device=p.device);other=1-obj;pi=p[ix,obj];pj=p[ix,other];si=shape[ix,obj];sj=shape[ix,other];zi=z[ix,obj];zj=z[ix,other];v=self.verb(verb);t=self.target(target);relative=(pj-pi)/scale
  delta=self.actor(torch.cat([pi/scale,si/.1,relative,sj/.1,zi[:,None],zj[:,None],v,t],-1));motion=delta.detach();radius=((pj-pi)/.1).square().sum(-1,keepdim=True);q=torch.cat([relative,radius,si/.1,sj/.1,zi[:,None],zj[:,None],v],-1);influence=torch.bmm(self.interaction(q).reshape(-1,3,3),motion[:,:,None]).squeeze(-1);out=p.clone();out[ix,obj]=pi+delta*scale;out[ix,other]=pj+influence*scale;return out

class StateActorDynamics(Dynamics):
 def __init__(self):
  nn.Module.__init__(self)
  self.held_update=nn.Sequential(nn.Linear(4,32),nn.SiLU(),nn.Linear(32,1))
  self.motion=StateActorMotion()
