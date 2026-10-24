"""Typed nodes and query-conditioned shared edge classifier. No colour/slot embedding."""
import torch
from torch import nn
class QueryRelationReader(nn.Module):
 def __init__(self):
  super().__init__()
  def enc(n):return nn.Sequential(nn.Linear(n,96),nn.LayerNorm(96),nn.GELU())
  self.block=enc(780);self.drawer=enc(1536);self.gripper=enc(1551)
  self.predicate=nn.Embedding(3,16)
  self.edge=nn.Sequential(nn.Linear(400,128),nn.LayerNorm(128),nn.GELU())
  self.history=nn.GRU(128,64,batch_first=True)
  self.answer=nn.Sequential(nn.Linear(64,32),nn.GELU(),nn.Linear(32,1))
 def nodes(self,objects,context,robot):
  blocks=self.block(objects)
  # Public static camera context anchors the drawer node; wrist+proprio anchors gripper.
  drawer=self.drawer(context[...,:1536]);gripper=self.gripper(torch.cat([context[...,1536:],robot],-1))
  return torch.cat([blocks,drawer[:,:,None],gripper[:,:,None]],2)
 def query(self,nodes,queries):
  # queries: [relation_type, source_node, destination_node]; 0Holding,1In,2Open.
  b,t,_,_=nodes.shape;q=queries.to(nodes.device);src=nodes[:,:,q[:,1]];dst=nodes[:,:,q[:,2]]
  rel=self.predicate(q[:,0])[None,None].expand(b,t,-1,-1)
  edge=self.edge(torch.cat([src,dst,src-dst,src*dst,rel],-1))
  _,h=self.history(edge.permute(0,2,1,3).reshape(b*len(q),t,128))
  return self.answer(h[-1]).reshape(b,len(q))
 def forward(self,objects,context,robot):
  q=torch.tensor([[0,j,4]for j in range(3)]+[[1,j,3]for j in range(3)]+[[2,3,3]],device=objects.device)
  return self.query(self.nodes(objects,context,robot),q)

def parse_query(text,entities):
 import re
 match=re.fullmatch(r'\s*(Holding|In|Open)\(([^()]*)\)\s*',text)
 if not match:raise ValueError('Supported templates: Holding(object), In(object, drawer), Open(drawer)')
 name,args=match.group(1),[a.strip()for a in match.group(2).split(',')]
 if name=='Holding'and len(args)==1 and args[0]in entities:return [0,entities[args[0]],4]
 if name=='In'and len(args)==2 and args[0]in entities and args[1]=='drawer':return [1,entities[args[0]],3]
 if name=='Open'and args==['drawer']:return [2,3,3]
 raise ValueError('Unknown entity or typed predicate arguments')
