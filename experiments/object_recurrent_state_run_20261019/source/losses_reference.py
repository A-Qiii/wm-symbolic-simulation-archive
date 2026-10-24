"""Common deterministic supervised objective. Definition only; no training loop."""
import torch
from models import history
from wm_symbolic.world_model import symbol_nll,outcome_loss

def loss_for_batch(model,batch,class_weights=None):
 o,g,v=batch['objects'],batch['context'],batch['availability'];verb,ptr,budget=batch['verb'],batch['pointer'],batch['budget'];step=batch['step_valid'].float();posts,priors=history(model,o,g,v,verb,ptr,budget)
 def sym(s,t):return symbol_nll(model.symbols(s),batch['cat'][:,t],batch['cat_valid'][:,t],batch['binary'][:,t],batch['binary_valid'][:,t],class_weights)
 def mean(vals,masks):
  val=torch.stack(vals,1);mask=torch.stack(masks,1);return(val*mask).sum()/mask.sum().clamp_min(1)
 cur=[sym(s,t)for t,s in enumerate(posts)];cm=[(step.sum(1)>0).float()]+[step[:,t]for t in range(step.shape[1])]
 prior=[];priorout=[];pm=[];multi=[];multiout=[];mm=[]
 for t,p in enumerate(priors):
  a=model.interface.action_features(verb[:,t],ptr[:,t],budget[:,t]);prior.append(sym(p,t+1));pm.append(step[:,t]);priorout.append(outcome_loss(model.outcomes(posts[t],p,a),batch['outcomes'][:,t],batch['events_valid'][:,t]))
 for start in range(step.shape[1]):
  state=posts[start];valid=torch.ones_like(step[:,0])
  for t in range(start,min(start+3,step.shape[1])):
   before=state;state,detail=model.imagine(state,verb[:,t],ptr[:,t],budget[:,t]);valid=valid*step[:,t];multi.append(sym(state,t+1));mm.append(valid);multiout.append(outcome_loss(model.outcomes(before,state,detail['action']),batch['outcomes'][:,t],batch['events_valid'][:,t]))
 length=step.sum(1).long();state=posts[2];terminal=[];tm=[]
 for t in range(2,step.shape[1]):
  state,_=model.imagine(state,verb[:,t],ptr[:,t],budget[:,t]);terminal.append(sym(state,t+1));tm.append(((length==t+1)&(length>2)).float())
 terms={'current':mean(cur,cm),'prior':mean(prior,pm),'multi3':mean(multi,mm),'terminal_from_h2':mean(terminal,tm),'outcome':.5*(mean(priorout,pm)+mean(multiout,mm))}
 total=terms['current']+.5*(terms['prior']+terms['multi3']+terms['terminal_from_h2']+terms['outcome'])
 return total,terms
