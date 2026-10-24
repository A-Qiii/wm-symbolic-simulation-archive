"""Loss and eight paired trajectory evaluation for the symbolic baseline."""
import torch
from wm_symbolic.world_model import State,symbol_nll,outcome_loss
from wm_symbolic.evaluation import mean_symbols

def state_from_h(h):
 return State(h,torch.zeros(*h.shape[:-1],1,1,device=h.device,dtype=h.dtype))

def loss_for_symbols(model,b,k,weights):
 # Select one unbiased posterior sample per item and keep its identity through time.
 samples=torch.randint(b['symbol_samples'].shape[2],(b['skill'].shape[0],),device=b['skill'].device)
 h=b['symbol_samples'][torch.arange(len(samples),device=samples.device),:,samples]
 actions=model.action(b['skill'],b['arguments'],b['budget'])
 def sym(state,t):return symbol_nll(model.symbols(state),b['cat'][:,t],b['cat_valid'][:,t],b['binary'][:,t],b['binary_valid'][:,t],weights)
 syms=[];outs=[];multi=[]
 for t in range(6):
  before=state_from_h(h[:,t]);after,_=model.imagine(before,actions[:,t]);syms.append(.5*(sym(before,t)+sym(after,t+1)))
  outs.append(outcome_loss(model.outcome_logits(before,actions[:,t],after),b['outcomes'][:,t],b['events_valid'][:,t]))
  if k:
   current=before
   for j in range(min(k,6-t)):
    step=t+j;previous=current;current,_=model.imagine(current,actions[:,step])
    out=outcome_loss(model.outcome_logits(previous,actions[:,step],current),b['outcomes'][:,step],b['events_valid'][:,step])
    multi.append(sym(current,step+1)+.2*out)
 terms={'sym':torch.stack(syms).mean(),'out':torch.stack(outs).mean(),'multi':torch.stack(multi).mean() if multi else syms[0].sum()*0}
 return terms['sym']+.5*terms['out']+.5*terms['multi'],terms

@torch.no_grad()
def evaluate_symbols(model,data,device='cuda'):
 training=model.training;model.eval();summary={k:0 for k in ['current_nll','prior_nll','event_nll','free_3_nll','free_5_nll','current_correct','prior_correct','current_facts','facts','changed_correct','changed_facts','steps','episodes']}
 for i in range(0,len(data.episodes),8):
  ids=list(range(i,min(i+8,len(data.episodes))));b=data.batch(ids,device);batch=len(ids);samples=b['symbol_samples'].shape[2]
  with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=device=='cuda'):
   actions=model.action(b['skill'],b['arguments'],b['budget']).repeat_interleave(samples,0)
   states=[state_from_h(b['symbol_samples'][:,t].reshape(batch*samples,32)) for t in range(7)];free=states[0]
   for t in range(6):
    prior,_=model.imagine(states[t],actions[:,t]);free,_=model.imagine(free,actions[:,t]);current=mean_symbols(model.symbols(states[t]),batch,samples);pred=mean_symbols(model.symbols(prior),batch,samples)
    for name,values,j in [('current',current,t),('prior',pred,t+1)]:
     summary[name+'_nll']+=float(symbol_nll(values,b['cat'][:,j],b['cat_valid'][:,j],b['binary'][:,j],b['binary_valid'][:,j]).sum())
     guess=torch.cat([values['objects'].argmax(-1),values['drawer'].argmax(-1)[:,None],values['slider'].argmax(-1)[:,None]],-1);valid=b['cat_valid'][:,j];correct=guess==b['cat'][:,j]
     summary[name+'_correct']+=int((valid&correct).sum());summary['current_facts' if name=='current' else 'facts']+=int(valid.sum())
     if name=='prior':
      changed=valid&b['cat_valid'][:,t]&(b['cat'][:,t]!=b['cat'][:,j]);summary['changed_correct']+=int((changed&correct).sum());summary['changed_facts']+=int(changed.sum())
    raw=model.outcome_logits(states[t],actions[:,t],prior).float().reshape(batch,samples,8);mean=torch.logit(raw.sigmoid().mean(1).clamp(1e-6,1-1e-6))
    summary['event_nll']+=float(outcome_loss(mean,b['outcomes'][:,t],b['events_valid'][:,t]).sum());summary['steps']+=batch
    if t+1 in [3,5]:summary[f'free_{t+1}_nll']+=float(symbol_nll(mean_symbols(model.symbols(free),batch,samples),b['cat'][:,t+1],b['cat_valid'][:,t+1],b['binary'][:,t+1],b['binary_valid'][:,t+1]).sum())
   summary['episodes']+=batch
 for name in ['current_nll','prior_nll','event_nll']:summary[name]/=max(summary['steps'],1)
 for name in ['free_3_nll','free_5_nll']:summary[name]/=max(summary['episodes'],1)
 for name,num,den in [('current_accuracy','current_correct','current_facts'),('prior_accuracy','prior_correct','facts'),('changed_accuracy','changed_correct','changed_facts')]:summary[name]=summary[num]/max(summary[den],1)
 summary.update(latent_samples=8,aggregation='Mean probabilities over eight paired cached perceptual states; deterministic symbolic transitions',accuracy_scope='Valid categorical facts; independent lights included in NLL')
 model.train(training);return summary
