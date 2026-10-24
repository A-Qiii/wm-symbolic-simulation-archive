"""Marginalize stochastic RSSM trajectories; never replace them by argmax latents."""
import torch
from wm_symbolic.world_model import symbol_nll,outcome_loss

def mean_symbols(pred,batch,samples):
 result={}
 for key,value in pred.items():
  value=value.float().reshape(batch,samples,*value.shape[1:])
  if key=='lights':result[key]=torch.logit(value.sigmoid().mean(1).clamp(1e-6,1-1e-6))
  else:result[key]=value.softmax(-1).mean(1).clamp_min(1e-9).log()
 return result

@torch.no_grad()
def evaluate_data(model,data,device='cuda',samples=8,limit=None):
 was_training=model.training;model.eval();summary={'current_nll':0.,'prior_nll':0.,'event_nll':0.,'free_3_nll':0.,'free_5_nll':0.,
  'current_correct':0,'prior_correct':0,'current_facts':0,'facts':0,'changed_correct':0,'changed_facts':0,'steps':0,'free_3_roots':0,'free_5_roots':0}
 cases=[];max_e=min(len(data.episodes),limit or len(data.episodes))
 with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
  torch.manual_seed(8139)
  for i in range(0,max_e,4):
   ids=list(range(i,min(i+4,max_e)));b=data.batch(ids,device);batch=len(ids)
   with torch.autocast('cuda',dtype=torch.bfloat16):
    encoded=model.encode(b['tokens'],b['geometry'],b['visible'],b['robot']);actions=model.action(b['skill'],b['arguments'],b['budget'])
    encoded=encoded.repeat_interleave(samples,0);actions=actions.repeat_interleave(samples,0)
    posts,_,priors,_=model.history(encoded,actions,deterministic=False);free=posts[0]
    for t in range(6):
     keep=b['step_valid'][:,t];summary['steps']+=int(keep.sum());cur=mean_symbols(model.symbols(posts[t]),batch,samples);pred=mean_symbols(model.symbols(priors[t]),batch,samples)
     free,_=model.imagine(free,actions[:,t],deterministic=False)
     for name,prediction,j in [('current_nll',cur,t),('prior_nll',pred,t+1)]:
      summary[name]+=float((symbol_nll(prediction,b['cat'][:,j],b['cat_valid'][:,j],b['binary'][:,j],b['binary_valid'][:,j])*keep).sum())
     raw=model.outcome_logits(posts[t],actions[:,t],priors[t]).float().reshape(batch,samples,1+model.event_count)
     outcome=torch.logit(raw.sigmoid().mean(1).clamp(1e-6,1-1e-6))
     summary['event_nll']+=float((outcome_loss(outcome,b['outcomes'][:,t],b['events_valid'][:,t])*keep).sum())
     for name,prediction,j in [('current',cur,t),('prior',pred,t+1)]:
      predicted=torch.cat([prediction['objects'].argmax(-1),prediction['drawer'].argmax(-1)[:,None],prediction['slider'].argmax(-1)[:,None]],-1)
      valid=b['cat_valid'][:,j]&keep[:,None];correct=predicted==b['cat'][:,j];summary[name+'_correct']+=int((correct&valid).sum())
      if name=='current':summary['current_facts']+=int(valid.sum())
      else:
       summary['facts']+=int(valid.sum());changed=valid&b['cat_valid'][:,t]&(b['cat'][:,t]!=b['cat'][:,j]);summary['changed_correct']+=int((correct&changed).sum());summary['changed_facts']+=int(changed.sum())
      for bi in range(batch):
       if len(cases)<60 and (valid[bi]&~correct[bi]).any():cases.append({'kind':name,'episode_id':data.episodes[ids[bi]]['episode_id'],'step':t,'prediction':predicted[bi].tolist(),'target':b['cat'][bi,j].tolist(),'valid':valid[bi].tolist()})
     if t+1 in [3,5]:
      prediction=mean_symbols(model.symbols(free),batch,samples);summary[f'free_{t+1}_nll']+=float((symbol_nll(prediction,b['cat'][:,t+1],b['cat_valid'][:,t+1],b['binary'][:,t+1],b['binary_valid'][:,t+1])*keep).sum());summary[f'free_{t+1}_roots']+=int(keep.sum())
 n=max(summary['steps'],1)
 for key in ['current_nll','prior_nll','event_nll']:summary[key]/=n
 for horizon in [3,5]:summary[f'free_{horizon}_nll']/=max(summary[f'free_{horizon}_roots'],1)
 summary['current_accuracy']=summary['current_correct']/max(summary['current_facts'],1);summary['prior_accuracy']=summary['prior_correct']/max(summary['facts'],1);summary['changed_accuracy']=summary['changed_correct']/max(summary['changed_facts'],1)
 summary.update(counterexamples=cases,latent_samples=samples,aggregation='Mean categorical/independent probabilities over paired stochastic trajectories',accuracy_scope='Valid categorical facts; independent lights are included in NLL')
 model.train(was_training);return summary
