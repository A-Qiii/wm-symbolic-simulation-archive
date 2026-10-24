import torch
from torch import nn
def mlp(a,h,b):return nn.Sequential(nn.Linear(a,h),nn.SiLU(),nn.Linear(h,b))
class Transition(nn.Module):
 def __init__(self):
  super().__init__();self.obj=mlp(64*3+4+1+1,128,64);self.scene=mlp(64*3+4+1,128,64)
 def forward(self,z,a):
  objs=z[:,:3];scene=z[:,3];pool=objs.mean(1);target=(objs*a[:,4:7,None]).sum(1);typ=a[:,:4];hold=a[:,7:8]
  dz=self.obj(torch.cat([objs,scene[:,None].expand(-1,3,-1),pool[:,None].expand(-1,3,-1),typ[:,None].expand(-1,3,-1),a[:,4:7,None],hold[:,None].expand(-1,3,-1)],-1))
  ds=self.scene(torch.cat([scene,pool,target,typ,hold],-1));return z+torch.cat([dz,ds[:,None]],1)
class Model(nn.Module):
 def __init__(self,norm,kind):
  super().__init__();self.kind=kind;self.transition=Transition()
  if kind=='generic':
   for k in ['object_mean','object_basis','scene_mean','scene_basis','generic_mean','generic_std']:self.register_buffer(k,torch.as_tensor(norm[k],dtype=torch.float32))
  else:
   self.object_encoder=mlp(268,128,64);self.scene_encoder=mlp(1039,256,64);self.context=mlp(192,128,64)
   self.object_decoder=mlp(64,128,268);self.scene_decoder=mlp(64,256,1039)
 def encode(self,obj,scene):
  if self.kind=='generic':return (torch.cat([(obj-self.object_mean)@self.object_basis,((scene-self.scene_mean)@self.scene_basis)[:,None]],1)-self.generic_mean)/self.generic_std
  o=self.object_encoder(obj);s=self.scene_encoder(scene);o=o+self.context(torch.cat([o,s[:,None].expand(-1,3,-1),o.mean(1)[:,None].expand(-1,3,-1)],-1));return torch.cat([o,s[:,None]],1)
 def decode(self,z):
  if self.kind=='generic':
   z=z*self.generic_std+self.generic_mean;return z[:,:3]@self.object_basis.T+self.object_mean,z[:,3]@self.scene_basis.T+self.scene_mean
  return self.object_decoder(z[:,:3]),self.scene_decoder(z[:,3])
 def freeze_representation(self):
  for n,p in self.named_parameters():
   if not n.startswith('transition.'):p.requires_grad_(False)
class Query(nn.Module):
 def __init__(self):
  super().__init__();self.net=mlp(64*3+5,128,1)
 def forward(self,z):
  b=len(z);scene=z[:,3];pool=z[:,:3].mean(1);outs=[]
  for typ in range(5):
   ids=range(3)if typ<2 else[-1]
   for j in ids:
    obj=z[:,j]if j>=0 else torch.zeros_like(scene);qt=torch.zeros(b,5,device=z.device);qt[:,typ]=1
    outs.append(self.net(torch.cat([obj,scene,pool,qt],-1)).squeeze(-1))
  return torch.stack(outs,-1)
def probs(logits):return torch.cat([logits[:,:6].sigmoid(),logits[:,6:].softmax(-1)],-1)
def feature_error(pred,obj,scene,before=None):
 po,ps=pred;eo=(po-obj).square();es=(ps-scene).square()
 if before is not None:
  bo,bs=before;eo*=1+4*(obj-bo).abs().clamp(max=1);es*=1+4*(scene-bs).abs().clamp(max=1)
 return torch.stack([eo[...,:256].mean((1,2)),eo[...,256:].mean((1,2)),es[:,:1024].mean(1),es[:,1024:].mean(1)],-1)
