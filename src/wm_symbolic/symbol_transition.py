"""Baseline dynamics receives only learned symbolic probabilities and known action.

The frozen perception RSSM is used for real observations outside this module.
No continuous latent vector, future image, or evaluator label is a dynamics input.
"""
import torch
from torch import nn
from wm_symbolic.world_model import State,mlp

class SymbolTransition(nn.Module):
 transition_source='learned_symbol_transition'
 def __init__(self,event_count=7):
  super().__init__();self.event_count=event_count
  self.skill=nn.Embedding(25,32);self.pointer=nn.Embedding(4,16);self.source=nn.Embedding(5,8);self.destination=nn.Embedding(5,8);self.executor=nn.Embedding(1,8);self.action_net=mlp(73,256)
  self.transition=mlp(32+256,32);self.outcome=mlp(32+256+32,1+event_count)
 def initialize_frozen_action_encoder(self,perception):
  for name in ['skill','pointer','source','destination','executor','action_net']:
   getattr(self,name).load_state_dict(getattr(perception,name).state_dict());getattr(self,name).requires_grad_(False)
 def action(self,skill,arguments,budget):
  return self.action_net(torch.cat([self.skill(skill),self.pointer(arguments[...,0]),self.source(arguments[...,1]),self.destination(arguments[...,2]),(budget/360)[...,None],self.executor(torch.zeros_like(skill))],-1))
 @staticmethod
 def from_symbols(pred):
  h=torch.cat([pred['objects'].flatten(-2),pred['drawer'],pred['slider'],pred['lights']],-1)
  return State(h,torch.zeros(*h.shape[:-1],1,1,device=h.device,dtype=h.dtype))
 @staticmethod
 def symbols(state):
  h=state.h
  if h.shape[-1]!=32:raise ValueError('Symbolic dynamics accepts only32 declared symbolic logits')
  return {'objects':h[...,:24].reshape(*h.shape[:-1],3,8),'drawer':h[...,24:27],'slider':h[...,27:30],'lights':h[...,30:32]}
 def probabilities(self,state):
  p=self.symbols(state)
  return torch.cat([p['objects'].float().softmax(-1).flatten(-2),p['drawer'].float().softmax(-1),p['slider'].float().softmax(-1),p['lights'].float().sigmoid()],-1)
 def imagine(self,state,action,deterministic=False):
  # Probabilistic symbolic belief is the complete recurrent state: exactly32 values.
  h=self.transition(torch.cat([self.probabilities(state).to(action.dtype),action],-1))
  return State(h,torch.zeros(*h.shape[:-1],1,1,device=h.device,dtype=h.dtype)),None
 def outcome_logits(self,before,action,after):
  return self.outcome(torch.cat([self.probabilities(before).to(action.dtype),action,self.probabilities(after).to(action.dtype)],-1))
