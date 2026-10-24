from common import *
import torch
from train import newmodel
from models import ObjectGRU,Interface,history
from data import load,batch
torch.set_num_threads(4)
checks=[]
for seed in [0,1]:
 a=newmodel('baseline',seed);b=newmodel('geometry',seed);assert statehash(a)==statehash(b)
 ba=batch(load('train',seed,'cpu','baseline'),[0,120]);bg=batch(load('train',seed,'cpu','geometry'),[0,120]);sa=a.initial(ba['objects'][:,0],ba['context'][:,0],ba['availability'][:,0]);sg=b.initial(bg['objects'][:,0],bg['context'][:,0],bg['availability'][:,0]);assert torch.equal(sa,sg)
 assert ba['objects'].shape[-1]==528 and ba['context'].shape[-1]==544
 # Explicit state isolation: future public features do not affect h0, and imagined successors take no observations.
 altered={k:v.clone()for k,v in bg.items()};altered['objects'][:,1:]+=100;altered['context'][:,1:]-=100
 pa,_=history(b,bg['objects'],bg['context'],bg['availability'],bg['verb'],bg['pointer'],bg['budget']);pb,_=history(b,altered['objects'],altered['context'],altered['availability'],altered['verb'],altered['pointer'],altered['budget']);assert torch.equal(pa[0],pb[0])
 checks.append({'seed':seed,'parameters_each':sum(p.numel()for p in a.parameters()),'shared_initial_hash':statehash(a),'initial_outputs_identical_despite_different_inputs':True,'future_observation_h0_isolation':True})
members=read(S/'configs/data_membership.json');assert sha(S/'configs/data_membership.json')==sha(O/'configs/data_membership.json')
for seed in [0,1]:assert sha(S/f'configs/schedule_seed{seed}.npz')==sha(O/f'configs/schedule_seed{seed}.npz')
train_ids={e['root_id']for e in read(S/'cache/train_metadata.json')};forbidden={e['root_id']for e in read(S/'cache/diagnosed_confirmation_metadata.json')};assert not(train_ids&forbidden)
save(S/'results/input_preflight.json',{'utc':now(),'checks':checks,'old_data_membership_exact':True,'old_schedule_bytes_exact':True,'new_confirmation_excluded_from_training':True,'geometry_both_seeds_identical':True,'untrained':True})
files=list((S/'source').glob('*.py'))+[S/'configs'/n for n in ['protocol.json','data_membership.json','episode_order.json','schedule_seed0.npz','schedule_seed1.npz','input_normalization.json']]
save(S/'configs/design_frozen.json',{'utc':now(),'sha256':{str(p):sha(p)for p in files},'training_updates_at_freeze':0})
for p in files+[S/'configs/design_frozen.json']:p.chmod(0o444)
print('DESIGN_FROZEN',checks,flush=True)
