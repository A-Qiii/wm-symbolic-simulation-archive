"""Algebraic batching of the approved common objective; no objective changes."""
import torch
from models import history
from wm_symbolic.world_model import symbol_nll,outcome_loss

def loss_for_batch(model,b,class_weights=None):
 posts,priors=history(model,b['objects'],b['context'],b['availability'],b['verb'],b['pointer'],b['budget'])
 step=b['step_valid'].float();B=step.shape[0];T=step.shape[1]
 def cat(xs):return torch.cat(xs,0)
 def sym(states,ts,masks):
  pred=model.symbols(cat(states));loss=symbol_nll(pred,cat([b['cat'][:,t]for t in ts]),cat([b['cat_valid'][:,t]for t in ts]),cat([b['binary'][:,t]for t in ts]),cat([b['binary_valid'][:,t]for t in ts]),class_weights);mask=cat(masks);return(loss*mask).sum()/mask.sum().clamp_min(1)
 cm=[(step.sum(1)>0).float()]+[step[:,t]for t in range(T)]
 current=sym(posts,list(range(T+1)),cm);prior=sym(priors,list(range(1,T+1)),[step[:,t]for t in range(T)])
 aa=model.interface.action_features(cat([b['verb'][:,t]for t in range(T)]),cat([b['pointer'][:,t]for t in range(T)]),cat([b['budget'][:,t]for t in range(T)]))
 ol=outcome_loss(model.outcomes(cat(posts[:-1]),cat(priors),aa),cat([b['outcomes'][:,t]for t in range(T)]),cat([b['events_valid'][:,t]for t in range(T)]));sm=cat([step[:,t]for t in range(T)]);po=(ol*sm).sum()/sm.sum().clamp_min(1)
 states=cat(posts[:-1]);valid=torch.ones(T*B,device=step.device);ms=[];mt=[];mm=[];outs=[]
 for k in range(3):
  ts=[min(t+k,T-1)for t in range(T)];valid=valid*cat([step[:,ts[t]]*(t+k<T)for t in range(T)])
  before=states;states,d=model.imagine(states,cat([b['verb'][:,t]for t in ts]),cat([b['pointer'][:,t]for t in ts]),cat([b['budget'][:,t]for t in ts]))
  ms.extend(states.split(B));mt.extend([t+1 for t in ts]);mm.extend(valid.split(B));out=outcome_loss(model.outcomes(before,states,d['action']),cat([b['outcomes'][:,t]for t in ts]),cat([b['events_valid'][:,t]for t in ts]));outs.append(out)
 multi=sym(ms,mt,mm);mask=cat(mm);mo=(cat(outs)*mask).sum()/mask.sum().clamp_min(1)
 length=step.sum(1).long();state=posts[2];tt=[];tm=[]
 for t in range(2,T):
  state,_=model.imagine(state,b['verb'][:,t],b['pointer'][:,t],b['budget'][:,t]);tt.append(state);tm.append(((length==t+1)&(length>2)).float())
 terminal=sym(tt,list(range(3,T+1)),tm)
 terms={'current':current,'prior':prior,'multi3':multi,'terminal_from_h2':terminal,'outcome':.5*(po+mo)}
 return current+.5*(prior+multi+terminal+terms['outcome']),terms
