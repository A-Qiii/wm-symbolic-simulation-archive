from pathlib import Path
import json
import numpy as np
import torch
from model import PredicateReader
P=Path('/home/robot/research/projects/wm_symbolic');E=P/'experiments/state_predicates_20261014'
def main():
 torch.set_num_threads(4);meta=json.loads((E/'data/metadata.json').read_text());p=np.load(E/'results/direct_predicates_seed0_probabilities.npy')
 with np.load(E/'data/inputs_targets.npz') as z:arrays={k:torch.tensor(z[k]) for k in ['objects','context','robot']};cat=z['cat'];split=z['splits'];near=z['near_negative']
 m=PredicateReader();m.load_state_dict(torch.load(E/'models/direct_predicates_seed0.pt',map_location='cpu',weights_only=False)['model']);m.eval();lookup={(r['episode_id'],r['step']):r['index'] for r in meta}
 chosen={}
 for i,r in enumerate(meta):
  if r['split']!='diagnostic_D' or r['step']==0:continue
  if cat[i,3]==0 and p[i,6]>=.5 and 'first_drawer_false_positive' not in chosen:chosen['first_drawer_false_positive']=(i,6)
  for j in range(3):
   if cat[i,j]!=6 and p[i,j]>=.5 and 'first_held_false_positive' not in chosen:chosen['first_held_false_positive']=(i,j)
 probes=[]
 for name,(i,col) in chosen.items():
  donor=lookup[meta[i]['episode_id'],meta[i]['step']-1]
  with torch.no_grad():base=m.probabilities(arrays['objects'][i:i+1],arrays['context'][i:i+1],arrays['robot'][i:i+1]);swap=m.probabilities(arrays['objects'][i:i+1],arrays['context'][i:i+1],arrays['robot'][donor:donor+1])
  probes.append({'case':name,'recipient_index':i,'donor_previous_boundary_index':donor,'predicate_index':col,'probability_original':float(base[0,col]),'probability_replace_only_robot_history':float(swap[0,col]),'all_original_probabilities':base[0].tolist(),'all_swap_probabilities':swap[0].tolist(),'recipient':meta[i],'donor':meta[donor],'interpretation_limit':'Off-manifold model sensitivity probe: unchanged current images, previous real proprioception. Demonstrates numerical dependence, not a physically executed counterfactual or proof that vision alone succeeds.'})
 contingency={}
 for s in ['train','tune','calibration','diagnostic_D']:
  mask=split==s;held=(cat[:,:3]==6).any(-1);opened=cat[:,3]==2
  contingency[s]={'empty_closed':int((mask&~held&~opened).sum()),'empty_open':int((mask&~held&opened).sum()),'held_closed':int((mask&held&~opened).sum()),'held_open':int((mask&held&opened).sum())}
 (E/'results/failure_diagnosis.json').write_text(json.dumps({'no_retraining_or_correction':True,'single_input_replacement':'Only 4-frame robot history is replaced; all visual inputs unchanged. Donor is immediately previous real boundary.','probes':probes,'training_state_contingencies':contingency,'near_negative_test_indices':np.flatnonzero((split=='diagnostic_D')&near.any(-1)).tolist()},ensure_ascii=False,indent=2))
 print(json.dumps({'probes':[{k:x[k] for k in ['case','recipient_index','donor_previous_boundary_index','probability_original','probability_replace_only_robot_history']} for x in probes],'contingencies':contingency},indent=2))
if __name__=='__main__':main()
