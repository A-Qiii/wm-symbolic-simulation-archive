"""Original forward and loss arithmetic; expose current-only loss for E routing."""
import torch
import torch.nn.functional as F
from wm_symbolic.world_model import symbol_nll,outcome_loss,kl
from common import partition

def losses(model,batch,k=0,class_weights=None):
 encoded=model.encode(batch['tokens'],batch['geometry'],batch['visible'],batch['robot'])
 actions=model.action(batch['skill'],batch['arguments'],batch['budget']);posts,qs,priors,ps=model.history(encoded,actions)
 mask=batch['step_valid'].float();denom=mask.sum().clamp_min(1)
 obs_loss=[];sym_loss=[];kl_loss=[];out_loss=[];current_sym=[];future_sym=[]
 def sym(state,t):return symbol_nll(model.symbols(state),batch['cat'][:,t],batch['cat_valid'][:,t],batch['binary'][:,t],batch['binary_valid'][:,t],class_weights)
 for t,prior in enumerate(priors):
  reconstruction=model.reconstruct(posts[t].vector())
  channels=F.smooth_l1_loss(reconstruction.float(),batch['target'][:,t].float(),reduction='none')
  obs_loss.append(.5*(channels[:,:-model.robot_dim].mean(-1)+channels[:,-model.robot_dim:].mean(-1)))
  cs,fs=sym(posts[t],t),sym(prior,t+1);current_sym.append(cs);future_sym.append(fs);sym_loss.append(.5*(cs+fs))
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
 current=terms['obs']+.5*mean(current_sym)
 future=.5*mean(future_sym)+.5*terms['kl']+.5*terms['out']+.5*terms['multi']
 return total,current,future,terms,encoded

def backward_routed(model,total,current,encoded,arm,divisor=2):
 """Two RSSM reverse passes, one encoder reverse pass. Same work in both arms."""
 e,r=partition(model)
 full=torch.autograd.grad(total/divisor,[encoded]+r,retain_graph=True)
 local=torch.autograd.grad(current/divisor,encoded)[0]
 incoming=full[0]if arm=='joint'else local
 assert arm in ['joint','route_current_only']
 encoded.backward(incoming)
 for p,g in zip(r,full[1:]):
  if p.grad is None:p.grad=g.detach().clone()
  else:p.grad.add_(g.detach())
 return {'encoded_full_norm':float(full[0].norm()),'encoded_current_norm':float(local.norm()),'encoded_difference_norm':float((full[0]-local).norm())}

def clip_groups(model):
 """Both arms clip E and R separately at1, avoiding cross-group norm coupling."""
 e,r=partition(model)
 a=torch.nn.utils.clip_grad_norm_(e,1.);b=torch.nn.utils.clip_grad_norm_(r,1.)
 assert torch.isfinite(a)and torch.isfinite(b)
 return float(a),float(b)
