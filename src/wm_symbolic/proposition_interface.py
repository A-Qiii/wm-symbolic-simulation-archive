"""Task propositions; no GT, action effects, or categorical-location certainty gate in new mode."""
import torch
from wm_symbolic.symbol_decoder import COLORS,LOCATIONS,truth,proposition_probability as old_probability

def probability(probs,events,name):
 for j,color in enumerate(COLORS):
  if name==color+'_table':return probs['objects'][...,j,:3].sum(-1)
  if name==color+'_slider':return probs['objects'][...,j,4:6].sum(-1)
 return old_probability(probs,events,name)

def required_facts(task):
 # Never hide a possible second held object by filtering down to goal objects.
 return tuple(sorted(set(task['dfa']['propositions'])|set(task['terminal_goals'])|{c+'_held' for c in COLORS}|{'hand_empty'}))

def action_observations(skill):
 # Observability only. False preconditions do NOT remove an action or invent its outcome.
 common={c+'_held' for c in COLORS}|{'hand_empty'}
 if skill.startswith('lift_'):return tuple(sorted(common|{skill.split('_')[1]+'_table'}))
 if skill=='place_in_drawer':return tuple(sorted(common|{'drawer_open'}))
 return tuple(sorted(common|{'drawer_open','drawer_closed'}))

def decode(probs,events,names,calibration,mode='proposition'):
 assert mode in ['legacy','proposition']
 if bool((probs['objects'][...,6]>=calibration.threshold).sum()>1):return {n:None for n in names},'multiple_held_conflict'
 facts={n:truth(probability(probs,events,n),calibration.event_threshold if n.endswith('_moved') or '_was_' in n else calibration.threshold) for n in names}
 # Both modes require individual held propositions wherever HandEmpty is used.
 # Thus an ambiguous sum of held margins cannot silently settle the hand identity.
 if 'hand_empty' in names and any(truth(probs['objects'][j,6],calibration.threshold) is None for j in range(3)):facts['hand_empty']=None
 if mode=='legacy':
  for name in names:
   for j,color in enumerate(COLORS):
    if name.startswith(color+'_') and not name.endswith('_moved') and float(probs['objects'][j].max())<calibration.threshold:facts[name]=None
   if name.startswith('drawer_') and float(probs['drawer'].max())<calibration.threshold:facts[name]=None
   if name.startswith('slider_') and float(probs['slider'].max())<calibration.threshold:facts[name]=None
   if name=='hand_empty' and bool((probs['objects'].max(-1).values<calibration.threshold).any()):facts[name]=None
 return facts,('unknown' if any(v is None for v in facts.values()) else None)

def goal_score(probs,goals,calibration):
 zeros=torch.zeros(*probs['lights'].shape[:-1],3,device=probs['lights'].device)
 return torch.stack([probability(probs,zeros,n) if v else 1-probability(probs,zeros,n) for n,v in goals.items()],-1).amin(-1)
