"""Shared observation/posterior/prior symbolic world model; no evaluator import."""
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F

@dataclass
class State:
 h:torch.Tensor
 z:torch.Tensor
 def vector(self):return torch.cat([self.h,self.z.flatten(-2)],-1)
 def detach(self):return State(self.h.detach(),self.z.detach())

def mlp(i,o,width=256):return nn.Sequential(nn.Linear(i,width),nn.LayerNorm(width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,o))

class WorldModel(nn.Module):
 def __init__(self,skill_count=25,robot_dim=15,event_count=3):
  super().__init__();self.robot_dim=robot_dim;self.event_count=event_count
  self.vision=nn.Linear(384,256);self.geometry=mlp(8,256);self.robot_token=mlp(robot_dim,256)
  layer=nn.TransformerEncoderLayer(256,4,dim_feedforward=1024,dropout=0.,batch_first=True,norm_first=True,activation='gelu')
  self.spatial=nn.TransformerEncoder(layer,2,enable_nested_tensor=False);self.observation=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,512))
  self.skill=nn.Embedding(skill_count,32);self.pointer=nn.Embedding(4,16);self.source=nn.Embedding(5,8);self.destination=nn.Embedding(5,8);self.executor=nn.Embedding(1,8)
  self.action_net=mlp(73,256)
  self.gru=nn.GRUCell(1024+256,512);self.prior=mlp(512,1024);self.posterior=mlp(1024,1024)
  self.object_query=nn.Embedding(3,32);self.object_symbol=mlp(1536+32,8);self.scene_symbol=mlp(1536,8)
  self.reconstruct=mlp(1536,2*16*384+robot_dim)
  self.outcome=mlp(1536+256+1536,1+event_count)
 def encode(self,tokens,geometry,valid,robot):
  lead=tokens.shape[:-2];n=tokens.shape[-2]
  x=self.vision(tokens.reshape(-1,n,384))+self.geometry(geometry.reshape(-1,n,8))
  robot_token=self.robot_token(robot.reshape(-1,self.robot_dim))[:,None]
  x=torch.cat([x,robot_token],1);mask=torch.cat([valid.reshape(-1,n),torch.ones(x.shape[0],1,device=x.device,dtype=torch.bool)],1)
  x=self.spatial(x,src_key_padding_mask=~mask)
  pooled=(x*mask[...,None]).sum(1)/mask.sum(1,keepdim=True)
  return self.observation(pooled).reshape(*lead,512)
 def action(self,skill,arguments,budget):
  # Arguments are static entity/source/destination IDs, never actual success.
  return self.action_net(torch.cat([self.skill(skill),self.pointer(arguments[...,0]),self.source(arguments[...,1]),self.destination(arguments[...,2]),
   (budget/360.)[...,None],self.executor(torch.zeros_like(skill))],-1))
 @staticmethod
 def probs(logits):
  p=F.softmax(logits.float().reshape(*logits.shape[:-1],32,32),-1);return .99*p+.01/32
 @staticmethod
 def sample(probs,deterministic=False):
  idx=probs.argmax(-1) if deterministic else torch.multinomial(probs.reshape(-1,32),1).reshape(probs.shape[:-1])
  hard=F.one_hot(idx,32).to(probs);return hard+probs-probs.detach()
 def initial(self,observation,deterministic=False):
  h=torch.zeros(*observation.shape[:-1],512,device=observation.device,dtype=observation.dtype)
  q=self.probs(self.posterior(torch.cat([h,observation],-1)));return State(h,self.sample(q,deterministic)),q
 def imagine(self,state,action,deterministic=False):
  h=self.gru(torch.cat([state.z.flatten(-2).to(action.dtype),action],-1),state.h.to(action.dtype))
  p=self.probs(self.prior(h));return State(h,self.sample(p,deterministic)),p
 def observe(self,predicted,observation,deterministic=False):
  q=self.probs(self.posterior(torch.cat([predicted.h,observation],-1)));return State(predicted.h,self.sample(q,deterministic)),q
 def symbols(self,state,object_ids=None):
  v=state.vector();ids=torch.arange(3,device=v.device) if object_ids is None else object_ids
  query=self.object_query(ids);q=query.expand(*v.shape[:-1],len(ids),32)
  objects=self.object_symbol(torch.cat([v.unsqueeze(-2).expand(*v.shape[:-1],len(ids),1536),q],-1));scene=self.scene_symbol(v)
  return {'objects':objects,'drawer':scene[...,:3],'slider':scene[...,3:6],'lights':scene[...,6:]}
 def outcome_logits(self,before,action,after):return self.outcome(torch.cat([before.vector(),action,after.vector()],-1))
 def history(self,encoded,actions,deterministic=False):
  root,q=self.initial(encoded[:,0],deterministic);posts=[root];qs=[q];priors=[];ps=[]
  for t in range(actions.shape[1]):
   prior,p=self.imagine(posts[-1],actions[:,t],deterministic);post,q=self.observe(prior,encoded[:,t+1],deterministic)
   priors.append(prior);ps.append(p);posts.append(post);qs.append(q)
  return posts,qs,priors,ps

def kl(q,p):return (q.float()*(q.float().log()-p.float().log())).sum((-1,-2))

def symbol_nll(pred,cat,cat_valid,binary,binary_valid,weights=None):
 losses=[];masks=[]
 for j in range(5):
  x=pred['objects'][...,j,:] if j<3 else pred['drawer' if j==3 else 'slider']
  weight=weights[j].to(x.device) if weights else None
  loss=F.cross_entropy(x.float(),cat[:,j],weight=weight,reduction='none');losses.append(loss);masks.append(cat_valid[:,j])
 binary_loss=F.binary_cross_entropy_with_logits(pred['lights'].float(),binary.float(),reduction='none')
 loss=torch.cat([torch.stack(losses,-1),binary_loss],-1);mask=torch.cat([torch.stack(masks,-1),binary_valid],-1).float()
 return (loss*mask).sum(-1)/mask.sum(-1).clamp_min(1)

def outcome_loss(logits,target,valid):
 loss=F.binary_cross_entropy_with_logits(logits.float(),target.float(),reduction='none')
 return loss[:,0]+(loss[:,1:]*valid.float()).sum(-1)/valid.sum(-1).clamp_min(1)

def loss_for_batch(model,batch,k=0,class_weights=None):
 encoded=model.encode(batch['tokens'],batch['geometry'],batch['visible'],batch['robot'])
 actions=model.action(batch['skill'],batch['arguments'],batch['budget']);posts,qs,priors,ps=model.history(encoded,actions)
 mask=batch['step_valid'].float();denom=mask.sum().clamp_min(1)
 obs_loss=[];sym_loss=[];kl_loss=[];out_loss=[]
 def sym(state,t):return symbol_nll(model.symbols(state),batch['cat'][:,t],batch['cat_valid'][:,t],batch['binary'][:,t],batch['binary_valid'][:,t],class_weights)
 for t,prior in enumerate(priors):
  reconstruction=model.reconstruct(posts[t].vector())
  channels=F.smooth_l1_loss(reconstruction.float(),batch['target'][:,t].float(),reduction='none')
  obs_loss.append(.5*(channels[:,:-model.robot_dim].mean(-1)+channels[:,-model.robot_dim:].mean(-1)))
  sym_loss.append(.5*(sym(posts[t],t)+sym(prior,t+1)))
  kl_loss.append(.8*kl(qs[t+1].detach(),ps[t]).clamp_min(1)+.2*kl(qs[t+1],ps[t].detach()).clamp_min(1))
  out_loss.append(outcome_loss(model.outcome_logits(posts[t],actions[:,t],prior),batch['outcomes'][:,t],batch['events_valid'][:,t]))
 def mean(xs):return (torch.stack(xs,1)*mask).sum()/denom
 terms={'obs':mean(obs_loss),'sym':mean(sym_loss),'kl':mean(kl_loss),'out':mean(out_loss)}
 mult=[];multmask=[]
 if k:
  for start in range(actions.shape[1]):
   current=posts[start];available=torch.ones(mask.shape[0],device=mask.device)
   for j in range(1,min(k,actions.shape[1]-start)+1):
    t=start+j-1;before=current;current,p=model.imagine(current,actions[:,t]);available=available*mask[:,t]
    out=outcome_loss(model.outcome_logits(before,actions[:,t],current),batch['outcomes'][:,t],batch['events_valid'][:,t])
    mult.append(sym(current,t+1)+.2*kl(qs[t+1].detach(),p)+.2*out);multmask.append(available)
  mm=torch.stack(multmask,1);terms['multi']=(torch.stack(mult,1)*mm).sum()/mm.sum().clamp_min(1)
 else:terms['multi']=terms['obs']*0
 total=terms['obs']+terms['sym']+.5*terms['kl']+.5*terms['out']+.5*terms['multi']
 return total,terms
