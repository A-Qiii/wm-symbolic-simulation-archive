"""New task weights. RGB state + own motion / learned conditional influence."""
import torch
from torch import nn
from torch.nn import functional as F
def mlp(n,m,width=128):return nn.Sequential(nn.Linear(n,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,m))
class RGBState(nn.Module):
 def __init__(self,width):
  super().__init__();self.encoder=nn.Sequential(nn.Linear(width,256),nn.SiLU(),nn.Linear(256,128),nn.SiLU());self.position=nn.Linear(128,3);self.extent=nn.Linear(128,3);self.held=nn.Linear(128,1);self.register_buffer('mu',torch.tensor([.1,-.14,.53]));self.register_buffer('sd',torch.tensor([.20,.10,.10]));self.region=mlp(3,3,32)
 def forward(self,x):
  h=self.encoder(x);p=self.position(h)*self.sd+self.mu;shape=F.softplus(self.extent(h))*.1;z=self.held(h).squeeze(-1);return h,p,shape,z
 def read_regions(self,p):return self.region((p-self.mu)/self.sd)
class RelativeTransition(nn.Module):
 def __init__(self,condition):
  super().__init__();self.condition=condition;self.verb=nn.Embedding(2,8);self.target=nn.Embedding(4,8);self.actor=mlp(3+3+128+128+2+16,3);self.interaction=mlp(20+(3 if condition=='direct_relative'else 0),3 if condition=='direct_relative'else 9)
  for layer in [self.actor[-1],self.interaction[-1]]:nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
 def forward(self,h,p,shape,z,verb,obj,target,scale):
  ix=torch.arange(len(p),device=p.device);other=1-obj;pi=p[ix,obj];pj=p[ix,other];si=shape[ix,obj];sj=shape[ix,other];hi=h[ix,obj];ci=h[:,2];zi=z[ix,obj];zj=z[ix,other];v=self.verb(verb);t=self.target(target)
  delta=self.actor(torch.cat([pi/scale,si/.1,hi,ci,zi[:,None],zj[:,None],v,t],-1));motion=delta.detach();relative=(pj-pi)/scale;radius=((pj-pi)/.1).square().sum(-1,keepdim=True);q=torch.cat([relative,radius,si/.1,sj/.1,zi[:,None],zj[:,None],v],-1)
  if self.condition=='direct_relative':influence=self.interaction(torch.cat([q,motion],-1))
  else:influence=torch.bmm(self.interaction(q).reshape(-1,3,3),motion[:,:,None]).squeeze(-1)
  out=p.clone();out[ix,obj]=pi+delta*scale;out[ix,other]=pj+influence*scale;return out
class Dynamics(nn.Module):
 def __init__(self,condition):
  super().__init__();self.held_update=nn.Sequential(nn.Linear(4,32),nn.SiLU(),nn.Linear(32,1));self.motion=RelativeTransition(condition)
 @staticmethod
 def pack(h,p,shape,z):
  out=torch.zeros(len(p),3,135,device=p.device,dtype=torch.float32);out[:,:2,:128]=h;out[:,2,:128]=h.mean(1);out[:,:2,128:131]=p;out[:,:2,131:134]=shape;out[:,:2,134]=z;return out
 def forward(self,state,verb,obj,target,scale):
  h=state[:,:,:128];p=state[:,:2,128:131];shape=state[:,:2,131:134];z=state[:,:2,134].sigmoid();nextp=self.motion(h,p,shape,z,verb,obj,target,scale);role=F.one_hot(obj,2).to(p.dtype);v=F.one_hot(verb,2).to(p.dtype)[:,None].expand(-1,2,-1);nextz=self.held_update(torch.cat([z[...,None],role[...,None],v],-1)).squeeze(-1);return self.pack(h[:,:2],nextp,shape,nextz)
