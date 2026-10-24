"""Current-state predicates from causal visible features; evaluator data forbidden."""
import torch
from torch import nn

class PredicateReader(nn.Module):
 def __init__(self,mode='direct_predicates'):
  super().__init__();self.mode=mode
  self.object_encoder=nn.Sequential(nn.Linear(780,128),nn.LayerNorm(128),nn.GELU())
  self.context_encoder=nn.Sequential(nn.Linear(3087,128),nn.LayerNorm(128),nn.GELU())
  self.object_history=nn.GRU(256,96,batch_first=True)
  self.scene_history=nn.GRU(128,96,batch_first=True)
  self.object_head=nn.Sequential(nn.Linear(96,64),nn.GELU(),nn.Linear(64,8 if mode=='matched_eight_class' else 2))
  self.drawer_head=nn.Sequential(nn.Linear(96,64),nn.GELU(),nn.Linear(64,3 if mode=='matched_eight_class' else 1))
 def forward(self,objects,context,robot):
  # Entity count is dynamic; no learned entity ID, simulator pose, or labels.
  b,t,k,_=objects.shape
  obj=self.object_encoder(objects);ctx=self.context_encoder(torch.cat([context,robot],-1))
  joint=torch.cat([obj,ctx[:,:,None].expand(-1,-1,k,-1)],-1).permute(0,2,1,3).reshape(b*k,t,256)
  _,h=self.object_history(joint);_,g=self.scene_history(ctx)
  return self.object_head(h[-1].reshape(b,k,96)),self.drawer_head(g[-1])
 def probabilities(self,objects,context,robot):
  o,d=self(objects,context,robot)
  if self.mode=='matched_eight_class':
   q=o.softmax(-1);out=torch.stack([q[:,:,6],q[:,:,3]],-1);drawer=d.softmax(-1)[:,2:3]
  else:out=o.sigmoid();drawer=d.sigmoid()
  return torch.cat([out[:,:,0],out[:,:,1],drawer],-1)

def targets(cat,valid):
 return torch.cat([(cat[:,:3]==6),(cat[:,:3]==3),(cat[:,3:4]==2)],-1).float(),torch.cat([valid[:,:3],valid[:,:3],valid[:,3:4]],-1)

def predicate_api(probabilities,thresholds,entity_names):
 """Separate predicate names from arbitrary caller entity handles; retains p."""
 k=len(entity_names);result=[]
 for i,name in enumerate(entity_names):
  for pred,j,t in [('held',i,thresholds[0]),('in',k+i,thresholds[1])]:
   p=float(probabilities[j]);result.append({'predicate':pred,'arguments':[name] if pred=='held' else [name,'drawer'],'probability':p,'value':True if p>=t else (False if p<=1-t else 'UNKNOWN')})
 p=float(probabilities[-1]);t=thresholds[2]
 result.append({'predicate':'open','arguments':['drawer'],'probability':p,'value':True if p>=t else (False if p<=1-t else 'UNKNOWN')})
 held=[r for r in result if r['predicate']=='held' and r['value'] is True]
 if len(held)>1:
  for r in held:r['value']='UNKNOWN';r['reason']='multiple_held_conflict'
 return result
