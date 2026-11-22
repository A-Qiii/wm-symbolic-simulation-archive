"""Supervised RGB states; exchangeable object dynamics. No support input.

This branch extends the explicit-state two-object control, not the latent-only
model. Object role in a task never enters these modules.
"""
import torch
from torch import nn
from torch.nn import functional as F
def mlp(n,m,w=128):return nn.Sequential(nn.Linear(n,w),nn.SiLU(),nn.Linear(w,w),nn.SiLU(),nn.Linear(w,m))
class RGBState(nn.Module):
 def __init__(self,width=1567):
  super().__init__();self.encoder=nn.Sequential(nn.Linear(width,256),nn.SiLU(),nn.Linear(256,128),nn.SiLU());self.position=nn.Linear(128,3);self.extent=nn.Linear(128,3);self.held=nn.Linear(128,1);self.region=mlp(6,3,32);self.register_buffer('mu',torch.tensor([.1,-.14,.53]));self.register_buffer('sd',torch.tensor([.20,.10,.10]))
 def forward(self,x):
  h=self.encoder(x);return h,self.position(h)*self.sd+self.mu,F.softplus(self.extent(h))*.1,self.held(h).squeeze(-1)
 def read_regions(self,p,size):return self.region(torch.cat([(p-self.mu)/self.sd,size/.1],-1))
class HeldHead(nn.Module):
 """Same per-object scoring function in BCE and categorical controls."""
 def __init__(self):
  super().__init__();self.score=nn.Linear(256,1);self.none=nn.Parameter(torch.zeros(()))
 def forward(self,h):
  c=h.mean(1,keepdim=True).expand_as(h);return torch.cat([self.none.expand(len(h),1),self.score(torch.cat([h,c],-1)).squeeze(-1)],-1)
 def probabilities(self,h,kind='ce'):
  z=self(h);return z.softmax(-1)[:,1:] if kind=='ce' else z[:,1:].sigmoid()
def pack(p,size,held):return torch.cat([p,size,held[...,None]],-1)
class ObjectDynamics(nn.Module):
 def __init__(self,kind):
  super().__init__();assert kind in ['diagonal','direct'];self.kind=kind;self.verb=nn.Embedding(2,8);self.target=nn.Embedding(4,8)
  self.context=mlp(20,32,64);self.actor=mlp(3+3+1+16+32+6,3)
  self.held_score=mlp(4,1,32);self.held_none=mlp(4,1,32)
  # Put differing module last to give shared layers identical initialization.
  self.effect=mlp(20+(3 if kind=='direct'else 0),3)
  for m in [self.actor[-1],self.effect[-1]]:nn.init.zeros_(m.weight);nn.init.zeros_(m.bias)
 def forward(self,state,verb,obj,target,scale):
  p=state[...,:3];size=state[...,3:6];held=state[...,6];b,n,_=state.shape;ix=torch.arange(b,device=p.device);pi=p[ix,obj];si=size[ix,obj];zi=held[ix,obj];v=self.verb(verb);t=self.target(target)
  relative=(p-pi[:,None])/scale;radius=((p-pi[:,None])/.1).square().sum(-1,keepdim=True)
  q=torch.cat([relative,radius,si[:,None].expand(-1,n,-1)/.1,size/.1,zi[:,None,None].expand(-1,n,1),held[...,None],v[:,None].expand(-1,n,-1)],-1)
  role=F.one_hot(obj,n).bool();other=(~role).float();context=(self.context(q)*other[...,None]).sum(1)/other.sum(1,keepdim=True)
  # Target "large_top" refers to the observed large-object anchor, index 0.
  # This is the action destination parameter, not a support/contact predicate.
  anchor=torch.cat([(p[:,0]-pi)/scale,size[:,0]/.1],-1)*(target==3)[:,None]
  delta=self.actor(torch.cat([pi/scale,si/.1,zi[:,None],v,t,context,anchor],-1));motion=delta.detach()
  effect=self.effect(torch.cat([q,motion[:,None].expand(-1,n,-1)],-1)) if self.kind=='direct'else self.effect(q)*motion[:,None]
  nextp=p+torch.where(role[...,None],delta[:,None],effect)*scale
  vh=F.one_hot(verb,2).float();scores=self.held_score(torch.cat([held[...,None],role[...,None].float(),vh[:,None].expand(-1,n,-1)],-1)).squeeze(-1)
  none=self.held_none(torch.cat([held.sum(1,keepdim=True),zi[:,None],vh],-1));nextheld=torch.cat([none,scores],-1).softmax(-1)[:,1:]
  return pack(nextp,size,nextheld)
