"""Shared categorical facts with explicit UNKNOWN; no action-effect rules."""
from dataclasses import dataclass
import torch
COLORS=('red','blue','pink')
LOCATIONS=('table_left','table_middle','table_right','drawer','slider_left','slider_right','held','other')
@dataclass
class Calibration:
 temperature:float=1.0
 threshold:float=.5
 event_threshold:float=.5
 calibrated:bool=False
 def as_dict(self):return dict(temperature=self.temperature,threshold=self.threshold,event_threshold=self.event_threshold,calibrated=self.calibrated)

def probabilities(pred,calibration):
 t=calibration.temperature
 return {k:(v.float()/t).sigmoid() if k=='lights' else (v.float()/t).softmax(-1) for k,v in pred.items()}

def proposition_probability(probs,event_probs,name):
 event_names=['red_moved','blue_moved','pink_moved','led_was_off','led_was_on','light_was_off','light_was_on']
 if name in event_names:
  j=event_names.index(name)
  return event_probs[...,j] if event_probs.shape[-1]>j else torch.full(event_probs.shape[:-1],float('nan'),device=event_probs.device)
 for j,color in enumerate(COLORS):
  if name.startswith(color+'_'):
   word=name[len(color)+1:]
   if word=='moved':return event_probs[...,j]
   if word in LOCATIONS:return probs['objects'][...,j,LOCATIONS.index(word)]
 if name in ['drawer_closed','drawer_middle','drawer_open']:return probs['drawer'][...,['drawer_closed','drawer_middle','drawer_open'].index(name)]
 if name in ['slider_right','slider_middle','slider_left']:return probs['slider'][...,['slider_right','slider_middle','slider_left'].index(name)]
 if name in ['light_on','led_on']:return probs['lights'][...,[ 'light_on','led_on'].index(name)]
 if name in ['light_off','led_off']:return 1-probs['lights'][...,[ 'light_off','led_off'].index(name)]
 raise KeyError('Unknown symbolic vocabulary: '+name)

def truth(prob,threshold):
 p=float(prob)
 if p>=threshold:return True
 if p<=1-threshold:return False
 return None

def decode(probs,event_probs,names,calibration):
 conflict=bool((probs['objects'][...,6]>=calibration.threshold).sum()>1)
 if conflict:return {n:None for n in names},'multiple_held_conflict'
 facts={n:truth(proposition_probability(probs,event_probs,n),calibration.event_threshold if n.endswith('_moved') or '_was_' in n else calibration.threshold) for n in names}
 return facts,('unknown' if any(v is None for v in facts.values()) else None)

def goal_score(probs,goals,calibration):
 zeros=torch.zeros(*probs['lights'].shape[:-1],3,device=probs['lights'].device)
 ps=[proposition_probability(probs,zeros,n) if wanted else 1-proposition_probability(probs,zeros,n) for n,wanted in goals.items()]
 return torch.stack(ps,-1).amin(-1)
