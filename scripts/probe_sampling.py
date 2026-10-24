from pathlib import Path
import json,fcntl
import torch
from wm_symbolic.world_model import WorldModel,symbol_nll
from wm_symbolic.dataset import Episodes
P=Path('/home/robot/research/projects/wm_symbolic');out=P/'runs/wm_probe_seed0'
lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
ck=torch.load(out/'checkpoint_01000.pt',map_location='cpu',weights_only=False);m=WorldModel().cuda();m.load_state_dict(ck['model']);d=Episodes(P,'train',128,ck['normalization']);result={}
@torch.no_grad()
def run(training,deterministic):
 m.train(training);correct=total=0;nll=0.;vectors=[]
 torch.manual_seed(911)
 for i in range(0,len(d.episodes),4):
  ids=list(range(i,min(i+4,len(d.episodes))));b=d.batch(ids,'cuda')
  with torch.autocast('cuda',dtype=torch.bfloat16):
   e=m.encode(b['tokens'],b['geometry'],b['visible'],b['robot']);vectors.append(e.float().cpu());a=m.action(b['skill'],b['arguments'],b['budget']);posts,_,priors,_=m.history(e,a,deterministic)
   for t,p in enumerate(priors):
    pred=m.symbols(p);y=torch.cat([pred['objects'].argmax(-1),pred['drawer'].argmax(-1)[:,None],pred['slider'].argmax(-1)[:,None]],-1);mask=b['cat_valid'][:,t+1]&b['step_valid'][:,t,None]
    correct+=int(((y==b['cat'][:,t+1])&mask).sum());total+=int(mask.sum());nll+=float((symbol_nll(pred,b['cat'][:,t+1],b['cat_valid'][:,t+1],b['binary'][:,t+1],b['binary_valid'][:,t+1])*b['step_valid'][:,t]).sum())
 return {'prior_categorical_accuracy':correct/total,'prior_nll':nll/d.transitions},vectors
vectors={}
for mode,training,det in [('train_sample',True,False),('eval_sample',False,False),('train_argmax',True,True),('eval_argmax',False,True)]:result[mode],vectors[mode]=run(training,det)
result['train_eval_encoder_max_abs_difference']=max(float((a-b).abs().max()) for a,b in zip(vectors['train_sample'],vectors['eval_sample']))
result['same_checkpoint_and_inputs']=True;result['purpose']='Identify whether replacing stochastic latent draws by marginal argmax changes the actual RSSM evaluation; no model/data edits.'
(out/'sampling_diagnostic.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
