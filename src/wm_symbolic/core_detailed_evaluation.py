"""Independent factual, persistence, and action-dependence evaluation."""
from collections import defaultdict
from wm_symbolic.core_field_metrics import FieldMetrics
import torch
from wm_symbolic.world_model import symbol_nll,outcome_loss
from wm_symbolic.evaluation import mean_symbols
from wm_symbolic.dataset import static_arguments

def cats(x):return torch.cat([x['objects'].argmax(-1),x['drawer'].argmax(-1)[:,None],x['slider'].argmax(-1)[:,None]],-1)
@torch.no_grad()
def evaluate(model,data,device='cuda',samples=8,limit=None,diagnostics=True):
 field_metrics=FieldMetrics();was=model.training;model.eval();s=defaultdict(float);byroot=defaultdict(lambda:defaultdict(float));cases=[];n=min(len(data.episodes),limit or len(data.episodes))
 def addmetric(name,pred,b,t,keep,roots,previous=None):
  field_metrics.add(name,pred,b,t,keep,previous,data.skills)
  valid=b['cat_valid'][:,t]&keep[:,None];correct=(cats(pred)==b['cat'][:,t])&valid
  loss=symbol_nll(pred,b['cat'][:,t],b['cat_valid'][:,t],b['binary'][:,t],b['binary_valid'][:,t])
  values={'correct':correct.sum(-1),'facts':valid.sum(-1),'nll_sum':loss*keep,'steps':keep}
  if previous is not None:
   changed=valid&b['cat_valid'][:,previous]&(b['cat'][:,t]!=b['cat'][:,previous]);values['changed_correct']=(correct&changed).sum(-1);values['changed_facts']=changed.sum(-1)
  for key,v in values.items():
   s[name+'_'+key]+=float(v.sum())
   for j,root in enumerate(roots):byroot[root][name+'_'+key]+=float(v[j])
  if name in ['prior','current'] and len(cases)<60:
   for j,root in enumerate(roots):
    if keep[j] and (valid[j]&~correct[j]).any():cases.append({'kind':name,'root':root,'boundary':t,'prediction':cats(pred)[j].tolist(),'target':b['cat'][j,t].tolist(),'valid':valid[j].tolist()})
 with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
  torch.manual_seed(81831)
  for i in range(0,n,4):
   ids=list(range(i,min(i+4,n)));b=data.batch(ids,device);batch=len(ids);roots=[data.episodes[j]['root_id'] for j in ids]
   with torch.autocast('cuda',dtype=torch.bfloat16):
    enc=model.encode(b['tokens'],b['geometry'],b['visible'],b['robot']);acts=model.action(b['skill'],b['arguments'],b['budget'])
    enc=enc.repeat_interleave(samples,0);acts=acts.repeat_interleave(samples,0);posts,_,priors,_=model.history(enc,acts)
    free=posts[0];root_symbols=mean_symbols(model.symbols(posts[0]),batch,samples)
    for t in range(4):
     keep=b['step_valid'][:,t];cur=mean_symbols(model.symbols(posts[t]),batch,samples);pr=mean_symbols(model.symbols(priors[t]),batch,samples)
     addmetric('current',cur,b,t,keep,roots);addmetric('prior',pr,b,t+1,keep,roots,t);addmetric('persistence',cur,b,t+1,keep,roots,t)
     valid=b['cat_valid'][:,t]&b['cat_valid'][:,t+1]&keep[:,None];same=b['cat'][:,t]==b['cat'][:,t+1]
     s['oracle_persistence_correct']+=int((same&valid).sum());s['oracle_persistence_facts']+=int(valid.sum())
     raw=model.outcome_logits(posts[t],acts[:,t],priors[t]).float().reshape(batch,samples,1+model.event_count);out=torch.logit(raw.sigmoid().mean(1).clamp(1e-6,1-1e-6))
     s['event_nll_sum']+=float((outcome_loss(out,b['outcomes'][:,t],b['events_valid'][:,t])*keep).sum());s['event_steps']+=int(keep.sum())
     free,_=model.imagine(free,acts[:,t]);fp=mean_symbols(model.symbols(free),batch,samples);addmetric('free'+str(t+1),fp,b,t+1,keep,roots,0)
     addmetric('free_persist'+str(t+1),root_symbols,b,t+1,keep,roots,0)
     if diagnostics:
      # A fixed derangement of all six requests; matching static arguments are changed too.
      mapping=torch.tensor([1,4,5,0,2,3],device=device);wrong=mapping[b['skill'][:,t]]
      args=torch.tensor([static_arguments(data.skills[j]) for j in wrong.tolist()],device=device);wa=model.action(wrong,args,b['budget'][:,t]).repeat_interleave(samples,0)
      # Pair stochastic samples so a difference cannot be attributed only to RNG.
      state_rng=torch.cuda.get_rng_state();wp,_=model.imagine(posts[t],wa);torch.cuda.set_rng_state(state_rng);cp,_=model.imagine(posts[t],acts[:,t])
      w=mean_symbols(model.symbols(wp),batch,samples);c=mean_symbols(model.symbols(cp),batch,samples);addmetric('wrong_action',w,b,t+1,keep,roots,t)
      delta=.5*(w['objects'].softmax(-1)-c['objects'].softmax(-1)).abs().sum(-1).mean(-1)
      s['action_object_total_variation_sum']+=float((delta*keep).sum());s['action_trials']+=int(keep.sum())
    keep=b['step_valid'][:,3];last=mean_symbols(model.symbols(posts[4]),batch,samples);addmetric('current',last,b,4,keep,roots)
 result=dict(s)
 for prefix in ['current','prior','persistence','wrong_action','free1','free2','free3','free4','free_persist1','free_persist2','free_persist3','free_persist4']:
  if prefix+'_steps' not in s:continue
  result[prefix+'_accuracy']=s[prefix+'_correct']/max(1,s[prefix+'_facts']);result[prefix+'_nll']=s[prefix+'_nll_sum']/max(1,s[prefix+'_steps'])
  if prefix+'_changed_facts' in s:result[prefix+'_changed_accuracy']=s[prefix+'_changed_correct']/max(1,s[prefix+'_changed_facts'])
 result['event_nll']=s['event_nll_sum']/max(1,s['event_steps']);result['oracle_persistence_accuracy']=s['oracle_persistence_correct']/max(1,s['oracle_persistence_facts'])
 result['action_object_total_variation']=s['action_object_total_variation_sum']/max(1,s['action_trials'])
 result.update(root_metrics={k:dict(v) for k,v in byroot.items()},counterexamples=cases,latent_samples=samples,episodes=n,roots=len(byroot),test_tuning=False,accuracy_scope='valid categorical facts; NLL also includes lights',horizon_scope='1-4 real requested skills, no padded steps',persistence_scope='model current-state marginal reused unchanged for next endpoint')
 result['detailed']=field_metrics.result()
 model.train(was);return result
