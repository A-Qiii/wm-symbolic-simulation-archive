"""Design prototype: deterministic GRUs, no optimizer and no training entry point.
Both arms receive EXACTLY the same frozen encoder object/context features.
Every prediction is computed from recurrent state; no observation-to-head bypass.
"""
import copy
import torch
from torch import nn

def mlp(i,o,w):return nn.Sequential(nn.Linear(i,w),nn.SiLU(),nn.Linear(w,o))
class Interface(nn.Module):
 def __init__(self):
  super().__init__();self.verb=nn.Embedding(4,16);self.action=mlp(18,32,64)
  self.bind_object=mlp(288,1,64);self.bind_null=mlp(160,1,64)
  self.object_head=mlp(256,8,128);self.scene_head=mlp(128,8,64)
  self.event_head=mlp(544,1,128);self.success_head=mlp(544,1,128)
 def action_features(self,verb,pointer,budget):
  return self.action(torch.cat([self.verb(verb),pointer.sum(-1,keepdim=True),(budget/900.)[...,None]],-1))
 def binding(self,nodes,a,verb,pointer):
  h,c=nodes[:,:3],nodes[:,3];cc=c[:,None].expand(-1,3,-1);aa=a[:,None].expand(-1,3,-1)
  logits=self.bind_object(torch.cat([h,cc,aa],-1)).squeeze(-1);null=self.bind_null(torch.cat([c,a],-1));soft=torch.cat([logits,null],-1).softmax(-1)[:,:3]
  # Requested lift pointer is public. Place (verb=3) infers its target from state.
  # Open/close target shared context. Every node still receives action/messages.
  return torch.where((verb==2)[:,None],pointer,torch.where((verb==3)[:,None],soft,torch.zeros_like(soft)))
 def symbols(self,nodes):
  h,c=nodes[:,:3],nodes[:,3];obj=self.object_head(torch.cat([h,c[:,None].expand(-1,3,-1)],-1));scene=self.scene_head(c)
  return {'objects':obj,'drawer':scene[:,:3],'slider':scene[:,3:6],'lights':scene[:,6:]}
 def outcomes(self,before,after,a):
  h,c=before[:,:3],before[:,3];hh,cc=after[:,:3],after[:,3]
  x=torch.cat([h,hh,c[:,None].expand(-1,3,-1),cc[:,None].expand(-1,3,-1),a[:,None].expand(-1,3,-1)],-1)
  success=self.success_head(torch.cat([h.mean(1),hh.mean(1),c,cc,a],-1));return torch.cat([success,self.event_head(x).squeeze(-1)],-1)
class ObjectGRU(nn.Module):
 def __init__(self,interface=None):
  super().__init__();self.interface=copy.deepcopy(interface)if interface is not None else Interface()
  self.prior_message=mlp(418,128,608);self.observe_message=mlp(1281,128,608)
  self.object_prior=nn.GRUCell(289,128);self.context_prior=nn.GRUCell(160,128)
  self.object_observe=nn.GRUCell(1153,128);self.context_observe=nn.GRUCell(640,128)
 @staticmethod
 def messages(fn,features):
  x=torch.stack([features(i,j)for i in range(3)for j in range(3)if j!=i],1);return fn(x).reshape(x.shape[0],3,2,-1).mean(2)
 def imagine(self,state,verb,pointer,budget):
  h,c=state[:,:3],state[:,3];a=self.interface.action_features(verb,pointer,budget);alpha=self.interface.binding(state,a,verb,pointer)
  m=self.messages(self.prior_message,lambda i,j:torch.cat([h[:,i],h[:,j],c,a,alpha[:,i:i+1],alpha[:,j:j+1]],-1))
  x=torch.cat([m,c[:,None].expand(-1,3,-1),a[:,None].expand(-1,3,-1),alpha[...,None]],-1)
  hn=self.object_prior(x.reshape(-1,289),h.reshape(-1,128)).reshape(-1,3,128);cn=self.context_prior(torch.cat([h.mean(1),a],-1),c)
  return torch.cat([hn,cn[:,None]],1),{'binding':alpha,'messages':m,'action':a}
 def observe(self,prior,objects,context,availability):
  h,c=prior[:,:3],prior[:,3]
  m=self.messages(self.observe_message,lambda i,j:torch.cat([h[:,i],h[:,j],objects[:,j],context,availability[:,j:j+1]],-1))
  x=torch.cat([objects,context[:,None].expand(-1,3,-1),m,availability[...,None]],-1)
  hn=self.object_observe(x.reshape(-1,1153),h.reshape(-1,128)).reshape(-1,3,128);cn=self.context_observe(torch.cat([context,hn.mean(1)],-1),c)
  return torch.cat([hn,cn[:,None]],1)
 def initial(self,objects,context,availability):return self.observe(objects.new_zeros(objects.shape[0],4,128),objects,context,availability)
 def symbols(self,state):return self.interface.symbols(state)
 def outcomes(self,before,after,a):return self.interface.outcomes(before,after,a)
class GlobalGRU(nn.Module):
 def __init__(self,interface=None):
  super().__init__();self.interface=copy.deepcopy(interface)if interface is not None else Interface()
  self.prior_input=nn.Linear(35,128);self.observe_input=nn.Linear(2051,128)
  self.prior=nn.GRUCell(128,512);self.observation=nn.GRUCell(128,512)
 def imagine(self,state,verb,pointer,budget):
  a=self.interface.action_features(verb,pointer,budget);alpha=self.interface.binding(state.reshape(-1,4,128),a,verb,pointer)
  return self.prior(self.prior_input(torch.cat([a,alpha],-1)),state),{'binding':alpha,'action':a}
 def observe(self,prior,objects,context,availability):
  x=torch.cat([objects.flatten(1),context,availability],-1);return self.observation(self.observe_input(x),prior)
 def initial(self,objects,context,availability):return self.observe(objects.new_zeros(objects.shape[0],512),objects,context,availability)
 def symbols(self,state):return self.interface.symbols(state.reshape(-1,4,128))
 def outcomes(self,before,after,a):return self.interface.outcomes(before.reshape(-1,4,128),after.reshape(-1,4,128),a)

def history(model,objects,context,availability,verbs,pointers,budgets):
 state=model.initial(objects[:,0],context[:,0],availability[:,0]);posts=[state];priors=[]
 for t in range(verbs.shape[1]):
  prior,_=model.imagine(state,verbs[:,t],pointers[:,t],budgets[:,t]);state=model.observe(prior,objects[:,t+1],context[:,t+1],availability[:,t+1]);priors.append(prior);posts.append(state)
 return posts,priors
