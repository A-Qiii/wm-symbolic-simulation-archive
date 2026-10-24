from pathlib import Path
import json,torch
from wm_symbolic.world_model import WorldModel
from wm_symbolic.dataset import Episodes
P=Path('/home/robot/research/projects/wm_symbolic');out=P/'runs/wm_probe_seed0';torch.set_num_threads(2)
ck=torch.load(out/'checkpoint_01000.pt',map_location='cpu',weights_only=False);m=WorldModel().eval();m.load_state_dict(ck['model']);d=Episodes(P,'train',128,ck['normalization']);rows=[]
with torch.inference_mode():
 for index in [0,len(d.episodes)//2,len(d.episodes)-1]:
  b=d.batch([index],'cpu');e=m.encode(b['tokens'],b['geometry'],b['visible'],b['robot']);a=m.action(b['skill'],b['arguments'],b['budget']);_,_,priors,_=m.history(e,a,True)
  perm=torch.randperm(b['tokens'].shape[-2]);ep=m.encode(b['tokens'][:,:,perm],b['geometry'][:,:,perm],b['visible'][:,:,perm],b['robot']);error=float((e-ep).abs().max());assert error<1e-5,error
  for t in range(len(d.episodes[index]['rows'])):
   corrupt=e.clone();corrupt[:,t+1:]=torch.randn_like(corrupt[:,t+1:])*100;_,_,changed,_=m.history(corrupt,a,True)
   difference=float((priors[t].vector()-changed[t].vector()).abs().max());assert difference==0
   rows.append({'episode_id':d.episodes[index]['episode_id'],'predicted_step':t+1,'future_observations_poisoned_from':t+1,'prior_max_abs_difference':difference,'fp32_slot_permutation_encoder_max_abs_difference':error})
(out/'prior_isolation_all_steps.json').write_text(json.dumps({'checks':rows,'passed':True,'device':'cpu','model_and_data_unchanged':True,'future_feedback_is_only_a_target':True},indent=2));print('PASSED',len(rows),'prior horizon checks and FP32 permutation invariance')
