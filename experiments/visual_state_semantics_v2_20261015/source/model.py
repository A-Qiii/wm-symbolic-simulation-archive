"""Frozen prior temporal encoder; genuinely new shared object-predicate query head."""
import torch,importlib.util
from torch import nn
from common import CAND
class QueryRelationReader(nn.Module):
 def __init__(self):
  super().__init__();spec=importlib.util.spec_from_file_location('pretrained_temporal_encoder_definition',CAND/'model.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);old=mod.PredicateReader();old.load_state_dict(torch.load(CAND/'reader_seed1.pt',map_location='cpu',weights_only=False)['model'])
  # Classification heads are not copied or called. Only encoders and GRUs transfer.
  self.object_encoder=old.object_encoder;self.context_encoder=old.context_encoder;self.object_history=old.object_history;self.scene_history=old.scene_history
  for module in [self.object_encoder,self.context_encoder,self.object_history,self.scene_history]:module.requires_grad_(False)
  self.predicate=nn.Embedding(5,16);self.edge=nn.Sequential(nn.Linear(400,128),nn.LayerNorm(128),nn.GELU(),nn.Linear(128,64),nn.GELU());self.answer=nn.Linear(64,1)
 def nodes(self,objects,context,robot):
  b,t,k,_=objects.shape
  with torch.no_grad():
   obj=self.object_encoder(objects);ctx=self.context_encoder(torch.cat([context,robot],-1));joint=torch.cat([obj,ctx[:,:,None].expand(-1,-1,k,-1)],-1).permute(0,2,1,3).reshape(b*k,t,256);_,h=self.object_history(joint);_,s=self.scene_history(ctx);h=h[-1].reshape(b,k,96);s=s[-1]
  # Both typed anchor nodes share public global scene context; role supplied by query.
  return torch.cat([h,s[:,None],s[:,None]],1)
 def logits_query(self,nodes,queries):
  q=queries.to(nodes.device);a=nodes[:,q[:,1]];b=nodes[:,q[:,2]];rel=self.predicate(q[:,0])[None].expand(len(nodes),-1,-1);return self.answer(self.edge(torch.cat([a,b,a-b,a*b,rel],-1))).squeeze(-1)
 def forward(self,objects,context,robot):
  q=torch.tensor([[0,j,4]for j in range(3)]+[[1,j,3]for j in range(3)]+[[j,3,3]for j in range(2,5)],device=objects.device);return self.logits_query(self.nodes(objects,context,robot),q)
 def probabilities(self,objects,context,robot):
  z=self(objects,context,robot);return torch.cat([z[:,:6].sigmoid(),z[:,6:].softmax(-1)],-1)
 def query(self,nodes,queries):
  raw=self.logits_query(nodes,queries).sigmoid();dq=torch.tensor([[j,3,3]for j in range(2,5)],device=nodes.device);drawer=self.logits_query(nodes,dq).softmax(-1);out=[]
  for k,row in enumerate(queries):out.append(drawer[:,int(row[0])-2]if int(row[0])>=2 else raw[:,k])
  return torch.stack(out,1)

def parse_query(text,entities):
 import re
 m=re.fullmatch(r'\s*(Holding|In|Closed|Open|Intermediate)\(([^()]*)\)\s*',text)
 if not m:raise ValueError('Finite templates only: Holding(object), In(object,drawer), Closed/Open/Intermediate(drawer)')
 name,args=m.group(1),[s.strip()for s in m.group(2).split(',')]
 if name=='Holding'and len(args)==1 and args[0]in entities:return [0,entities[args[0]],4]
 if name=='In'and len(args)==2 and args[0]in entities and args[1]=='drawer':return [1,entities[args[0]],3]
 if name in ['Closed','Open','Intermediate']and args==['drawer']:return [2+['Closed','Open','Intermediate'].index(name),3,3]
 raise ValueError('Unknown entity or invalid typed query')

def decode(p):
 import numpy as np
 p=np.asarray(p);y=(p>=.5).astype(int);y[:,6:]=np.eye(3,dtype=int)[p[:,6:].argmax(-1)];h=y[:,:3];y[:,:3]=np.where((h.sum(-1)>1)[:,None]&(h==1),-1,h);return y
