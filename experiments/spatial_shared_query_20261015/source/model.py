"""Shared typed query network; only pool resolution differs between the two arms."""
from common import *
from baseline_model import QueryRelationReader as TemporalReader,parse_query,decode
import torch
from torch import nn
class QueryRelationReader(TemporalReader):
 def __init__(self,variant):
  assert variant in ['pooled','spatial'];super().__init__();self.variant=variant
  cp=read(BASE/'configs/reader_frozen.json');assert sha(cp['path'])==cp['sha256'];self.load_state_dict(torch.load(cp['path'],map_location='cpu',weights_only=False)['model'])
  prior=self.edge[0];expanded=nn.Linear(784,128)
  with torch.no_grad():
   expanded.weight.zero_();expanded.bias.copy_(prior.bias)
   for block in range(4):expanded.weight[:,block*192:block*192+96].copy_(prior.weight[:,block*96:block*96+96])
   expanded.weight[:,768:].copy_(prior.weight[:,384:])
  self.edge[0]=expanded
  self.patch_projection=nn.Sequential(nn.Linear(384,32),nn.LayerNorm(32),nn.GELU())
  self.spatial_object=nn.Sequential(nn.Linear(268,96),nn.LayerNorm(96),nn.GELU())
  self.spatial_scene=nn.Sequential(nn.Linear(1039,96),nn.LayerNorm(96),nn.GELU())
 def nodes(self,objects,context,robot,local,grid,bbox):
  old=super().nodes(objects,context,robot);b=len(objects)
  if self.variant=='pooled':
   local=local.mean(3,keepdim=True).expand(-1,-1,-1,4,-1)
   grid=grid.reshape(b,2,2,2,2,2,384).mean((3,5)).repeat_interleave(2,2).repeat_interleave(2,3).reshape(b,2,16,384)
  # Camera and row-major patch positions remain explicit through flattening.
  o=self.patch_projection(local).permute(0,2,1,3,4).reshape(b,3,256)
  o=self.spatial_object(torch.cat([o,bbox],-1))
  s=self.spatial_scene(torch.cat([self.patch_projection(grid).reshape(b,1024),robot[:,-1]],-1))
  extra=torch.cat([o,s[:,None],s[:,None]],1)
  return torch.cat([old,extra],-1)
 def forward(self,**x):
  nodes=self.nodes(**x);q=torch.tensor([[0,j,4]for j in range(3)]+[[1,j,3]for j in range(3)]+[[j,3,3]for j in range(2,5)],device=nodes.device);return self.logits_query(nodes,q)
 def probabilities(self,**x):
  z=self(**x);return torch.cat([z[:,:6].sigmoid(),z[:,6:].softmax(-1)],-1)
