"""Strict forward interface: current hidden state and declared action ONLY."""
import torch
from torch import nn
class LatentTransition(nn.Module):
 def __init__(self,mode='full'):
  super().__init__();assert mode in ['full','action_only','state_only'];self.mode=mode;self.object_update=nn.Sequential(nn.Linear(199,128),nn.GELU(),nn.Linear(128,96));self.scene_update=nn.Sequential(nn.Linear(201,128),nn.GELU(),nn.Linear(128,96))
  for net in [self.object_update,self.scene_update]:nn.init.zeros_(net[-1].weight);nn.init.zeros_(net[-1].bias)
 def forward(self,current_objects,current_scene,declared_actions):
  assert current_objects.shape[1:]==(3,96)and current_scene.shape[1:]==(96,)and declared_actions.shape[1:]==(9,)
  h,s,a=current_objects,current_scene,declared_actions
  if self.mode=='action_only':h=torch.zeros_like(h);s=torch.zeros_like(s)
  if self.mode=='state_only':a=torch.zeros_like(a)
  b=h.shape[0];kind=a[:,:4,None].transpose(1,2).expand(-1,3,-1);target_match=a[:,4:7,None];budget=a[:,None,7:9].expand(-1,3,-1);obj=torch.cat([h,s[:,None].expand(-1,3,-1),kind,target_match,budget],-1);scene=torch.cat([s,h.mean(1),a],-1)
  return h+self.object_update(obj),s+self.scene_update(scene)
