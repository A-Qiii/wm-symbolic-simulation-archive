"""Read-only ABC inference and paired proposition decoding, no optimizer."""
from pathlib import Path
import json,hashlib,datetime
from collections import Counter,defaultdict
import numpy as np
import torch
from wm_symbolic.world_model import WorldModel
from wm_symbolic.core_dataset import CoreEpisodes
from wm_symbolic.gpu_phase import gpu_phase
from wm_symbolic.symbol_decoder import Calibration,probabilities,event_probabilities
from wm_symbolic.proposition_interface import decode,probability,required_facts,action_observations
P=Path('/home/robot/research/projects/wm_symbolic');R=P/'runs/proposition_interface_20261014';OLD=P/'runs/core1200_v1';COLORS=['red','blue','pink']
NAMES=tuple([c+'_'+n for c in COLORS for n in ['table','drawer','held','moved']]+['drawer_closed','drawer_open','slider_right','hand_empty'])
def target(cat,valid,events,name):
 if name=='hand_empty':return not any(x==6 for x in cat[:3]) if all(valid[:3]) else None
 if name=='slider_right':return cat[4]==0 if valid[4] else None
 if name.startswith('drawer_'):return cat[3]==(0 if name=='drawer_closed' else 2) if valid[3] else None
 c,w=name.split('_',1);j=COLORS.index(c)
 if w=='moved':return bool(events[j])
 if not valid[j]:return None
 return cat[j] in [0,1,2] if w=='table' else cat[j]=={'drawer':3,'held':6}[w]
def collect():
 ck=torch.load(OLD/'training_main/best.pt',map_location='cpu',weights_only=False);base=json.loads((OLD/'evaluation/calibration/calibration.json').read_text());cal=Calibration(**base['calibration']);records=[]
 with gpu_phase(P,'interface_ABC_readonly_inference'),torch.inference_mode():
  model=WorldModel(skill_count=6,event_count=3).cuda().eval();model.load_state_dict(ck['model']);torch.manual_seed(72833);torch.set_num_threads(4)
  for split in ['train','dev']:
   data=CoreEpisodes(P,split,normalizer=ck['normalization'])
   for i,episode in enumerate(data.episodes):
    b=data.batch([i],'cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16):
     enc=model.encode(b['tokens'],b['geometry'],b['visible'],b['robot']).repeat_interleave(8,0);act=model.action(b['skill'],b['arguments'],b['budget']).repeat_interleave(8,0);posts,_,priors,_=model.history(enc,act)
     eventlist=[torch.zeros(8,3,device='cuda')]+[event_probabilities(model.outcome_logits(posts[t],act[:,t],priors[t])[:,1:],cal) for t in range(4)]
     for kind,seq in [('current',posts[:5]),('prior',priors[:4])]:
      for j,state in enumerate(seq):
       t=j+(kind=='prior');probs=probabilities(model.symbols(state),cal);truths={n:target(b['cat'][0,t].tolist(),b['cat_valid'][0,t].tolist(),([0,0,0] if t==0 else b['outcomes'][0,t-1,1:].tolist()),n) for n in NAMES}
       root=episode['root_id'];cohort='AB_in_sample' if split=='train' else 'C_calibration' if int(root.rsplit('_',1)[-1])>=12 else 'C_development'
       records.append({'root':root,'episode':i,'boundary':t,'kind':kind,'cohort':cohort,'probs':{k:v.cpu() for k,v in probs.items()},'events':eventlist[t].cpu(),'truths':truths})
    if i%30==0:print('ABC_INFERENCE',split,i,len(data.episodes),flush=True)
  del model;torch.cuda.empty_cache()
 torch.save(records,R/'evaluation/ABC_probabilities.pt');return records,base
def counts(records,tau,etau):
 cal=Calibration(threshold=tau,event_threshold=etau);tot=defaultdict(Counter);byprop=defaultdict(Counter);paired=defaultdict(Counter);claims=Counter();claimroots=set()
 for r in records:
  p={k:v.mean(0) for k,v in r['probs'].items()};e=r['events'].mean(0);co=r['cohort'];dec={mode:decode(p,e,NAMES,cal,mode)[0] for mode in ['legacy','proposition']}
  for mode,f in dec.items():
   for n in NAMES:
    truth=r['truths'][n]
    if truth is None:continue
    c=tot[(co,r['kind'],mode)];v=byprop[(co,r['kind'],mode,n)];key='unknown' if f[n] is None else 'correct' if f[n]==truth else 'wrong'
    c['valid']+=1;c[key]+=1;v['valid']+=1;v[key]+=1
  for n in NAMES:
   if r['truths'][n] is None:continue
   if dec['legacy'][n] is None and dec['proposition'][n] is not None:paired[(co,r['kind'])]['rescued_correct' if dec['proposition'][n]==r['truths'][n] else 'rescued_wrong']+=1
  if co=='C_calibration':
   for color in COLORS:
    for loc in ['drawer','held']:
     for state in ['closed','open']:
      ns=[color+'_'+loc,'drawer_'+state];f=dec['proposition']
      if all(r['truths'][n] is not None for n in ns):
       claims['eligible']+=1;actual=all(r['truths'][n] for n in ns);claims['true_combinations']+=actual
       if all(f[n] is True for n in ns):claims['claims']+=1;claims['false']+=not actual;claims['true_claims']+=actual;claimroots.add(r['root'])
 return {'rows':[dict(cohort=k[0],kind=k[1],mode=k[2],**v) for k,v in tot.items()],'per_proposition':[dict(cohort=k[0],kind=k[1],mode=k[2],name=k[3],**v) for k,v in byprop.items()],'paired_rescue':[dict(cohort=k[0],kind=k[1],**v) for k,v in paired.items()],'C_terminal_claims':dict(claims,claim_roots=len(claimroots))}
def main():
 assert not (R/'evaluation/threshold_freeze.json').exists()
 records,base=collect();curve=[]
 for tau in [.8,.85,.9,.95]:
  for etau in [.9,.95,.99]:
   result=counts([r for r in records if r['cohort']=='C_calibration'],tau,etau);v=[x for x in result['rows'] if x['mode']=='proposition'];wrong=sum(x.get('wrong',0) for x in v);known=sum(x.get('wrong',0)+x.get('correct',0) for x in v);cl=result['C_terminal_claims'];eligible=known>0 and wrong/known<=.05 and cl.get('claims',0)>0 and cl.get('false',0)/cl['claims']<=.05
   curve.append({'tau':tau,'event_tau':etau,'wrong_assertions':wrong,'known_assertions':known,'assertion_error_rate':wrong/max(1,known),'terminal_claims':cl,'eligible':eligible})
 chosen=next((x for x in curve if x['eligible']),None);tau,etau=(chosen['tau'],chosen['event_tau']) if chosen else (.8,.9)
 cal=dict(base['calibration'],threshold=tau,event_threshold=etau,calibrated=False)
 frozen={'calibration':cal,'checkpoint_sha256':base['checkpoint_sha256'],'selected_on':'Existing independent eight C calibration roots only; no D readings','empirical_grid_criterion_met':chosen is not None,'temperature_refit':False,'threshold_curve':curve,'completion_certified':False,'reason':'Small correlated calibration cohort and no full LTL process certification;10-root/30-claim rules are previous engineering additions, not original-guide requirements.','frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 (R/'evaluation/threshold_freeze.json').write_text(json.dumps(frozen,indent=2));(R/'evaluation/threshold_freeze.json').chmod(0o444)
 result=counts(records,tau,etau);result.update(thresholds=cal,independent_roots={'AB_in_sample':60,'C_development':12,'C_calibration':8},sample_unit='current5 boundaries and prior4 endpoints per four-skill branch; three branches/root. Same eight latent samples reused by both decoders.',unnecessary_UNKNOWN='Legacy UNKNOWN that proposition decoder makes definite; report rescued correct and rescued wrong separately against offline actual labels. Wrong rescues are not called unnecessary UNKNOWN.',test_used=False)
 (R/'evaluation/ABC_paired_metrics.json').write_text(json.dumps(result,indent=2));print(json.dumps({'selected_threshold':tau,'event_threshold':etau,'criterion_met':chosen is not None,'rows':result['rows'],'paired_rescue':result['paired_rescue']},indent=2))
if __name__=='__main__':main()
