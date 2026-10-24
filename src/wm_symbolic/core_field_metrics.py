"""Read-only summaries of existing predictions; no extra model calls or RNG."""
from collections import defaultdict,Counter
import torch
NAMES=['red_location','blue_location','pink_location','drawer_state','slider_state','lightbulb','led']
class FieldMetrics:
 def __init__(self):
  self.fields=defaultdict(lambda:defaultdict(Counter));self.states=defaultdict(Counter);self.skills=defaultdict(lambda:defaultdict(Counter));self.confusion=defaultdict(lambda:defaultdict(Counter))
 def add(self,name,pred,b,t,keep,previous,skills):
  predicted=torch.cat([pred['objects'].argmax(-1),pred['drawer'].argmax(-1)[:,None],pred['slider'].argmax(-1)[:,None],(pred['lights'].sigmoid()>.5).long()],-1)
  actual=torch.cat([b['cat'][:,t],b['binary'][:,t].long()],-1);valid=torch.cat([b['cat_valid'][:,t],b['binary_valid'][:,t]],-1)&keep[:,None];correct=predicted==actual
  old=None;oldvalid=None
  if previous is not None:
   old=torch.cat([b['cat'][:,previous],b['binary'][:,previous].long()],-1);oldvalid=torch.cat([b['cat_valid'][:,previous],b['binary_valid'][:,previous]],-1)
  for j,field in enumerate(NAMES):
   c=self.fields[name][field];c['correct']+=int((correct[:,j]&valid[:,j]).sum());c['valid']+=int(valid[:,j].sum())
   if old is not None:
    changed=valid[:,j]&oldvalid[:,j]&(old[:,j]!=actual[:,j]);unchanged=valid[:,j]&oldvalid[:,j]&(old[:,j]==actual[:,j])
    for tag,mask in [('changed',changed),('unchanged',unchanged)]:c[tag+'_correct']+=int((correct[:,j]&mask).sum());c[tag+'_valid']+=int(mask.sum())
   if name in ['current','prior','free4']:
    for i in range(len(keep)):
     if valid[i,j]:self.confusion[name][field][f'{int(actual[i,j])}->{int(predicted[i,j])}']+=1
  for width in [5,7]:
   v=valid[:,:width].all(-1);c=correct[:,:width].all(-1)&v;self.states[name][f'all{width}_correct']+=int(c.sum());self.states[name][f'all{width}_valid']+=int(v.sum())
  hv=valid[:,:3].all(-1);hc=((predicted[:,:3]==6)==(actual[:,:3]==6)).all(-1)&hv
  self.states[name]['holding_identity_correct']+=int(hc.sum());self.states[name]['holding_identity_valid']+=int(hv.sum())
  if name in ['prior','persistence','wrong_action']:
   for i in range(len(keep)):
    if not keep[i]:continue
    skill=skills[int(b['skill'][i,t-1])];c=self.skills[name][skill];v=valid[i,:5];c['correct']+=int((correct[i,:5]&v).sum());c['valid']+=int(v.sum())
    if old is not None:
     for tag,mask in [('changed',v&oldvalid[i,:5]&(old[i,:5]!=actual[i,:5])),('unchanged',v&oldvalid[i,:5]&(old[i,:5]==actual[i,:5]))]:c[tag+'_correct']+=int((correct[i,:5]&mask).sum());c[tag+'_valid']+=int(mask.sum())
 def result(self):
  def ratios(c):
   x=dict(c)
   for k in list(x):
    if k.endswith('correct'):
     prefix=k[:-7];den=x.get(prefix+'valid',0);x[prefix+'accuracy']=x[k]/den if den else None
   return x
  return {'field_metrics':{n:{k:ratios(v) for k,v in fields.items()} for n,fields in self.fields.items()},'state_metrics':{n:ratios(v) for n,v in self.states.items()},'skill_metrics':{n:{k:ratios(v) for k,v in skills.items()} for n,skills in self.skills.items()},'confusion_counts':{n:{k:dict(v) for k,v in fields.items()} for n,fields in self.confusion.items()},'definitions':{'main_accuracy':'sum correct / sum valid over5 categorical fields; not complete-state accuracy; lights excluded from this percentage','all5':'All three object-location fields plus drawer and slider correct at the same boundary; only boundaries with all5 labels valid','all7':'All5 categorical fields plus both binary lights correct; only fully valid boundaries','changed':'Actual before and after differ; both labels valid. Does not mean requested change.','unchanged':'Actual before and after equal; both labels valid.','holding_identity':'Exact set of held object identities from three object categories; all three labels valid','wrong_action':'Fixed derangement: open->close, close->lift_pink, lift_red->place, lift_blue->open, lift_pink->lift_red, place->lift_blue. Static arguments changed consistently;900 budget fixed. Score still uses the ORIGINAL requested action actual endpoint: this checks action dependence, not accuracy of the alternative action counterfactual. No executor call or zero-action token.','persistence':'Reuses the learned current-state distribution unchanged; no dynamics query.','free_persistence':'Reuses the same root distribution for every horizon; no intermediate true observations.'}}
