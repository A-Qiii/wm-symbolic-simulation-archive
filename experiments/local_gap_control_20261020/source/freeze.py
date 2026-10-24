from common import *
import torch,ast
from train import newmodel
from data import load
torch.set_num_threads(4);old_ns={'np':np,'torch':torch,'read':read,'S':OLD,'O':O,'C':C}
tree=ast.parse((OLD/'source/data.py').read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in ['source_cache','raw_inputs','load']];exec(compile(ast.Module(body=nodes,type_ignores=[]),'original geometry loader','exec'),old_ns)
checks=[]
for seed in [0,1]:
 for split in ['train','development']:
  a=load(split,seed,'cpu','coarse');b=load(split,seed,'cpu','local');original=old_ns['load'](split,seed,'cpu','geometry');assert all(torch.equal(a[k],original[k])for k in a),'coarse input changed'
  assert all(torch.equal(a[k],b[k])for k in a if k!='context');assert torch.equal(a['context'][...,:512],b['context'][...,:512]);checks.append({'seed':seed,'split':split,'coarse_bit_identical_to_previous_geometry':True,'all_other_inputs_bit_identical_between_arms':True})
 a=newmodel('coarse',seed);b=newmodel('local',seed);assert statehash(a)==statehash(b);assert sum(p.numel()for p in a.parameters())==2490068
 save(S/f'results/initialization_seed{seed}.json',{'sha256':statehash(a),'parameters_each':2490068,'same_between_arms':True})
for seed in [0,1]:assert sha(S/f'configs/schedule_seed{seed}.npz')==sha(O/f'configs/schedule_seed{seed}.npz')
save(S/'results/input_preflight.json',{'utc':now(),'checks':checks,'only_32_context_dimensions_differ':True,'old_data_schedules_identical':True,'training_updates':0})
files=list((S/'source').glob('*.py'))+[S/'configs'/n for n in ['protocol.json','data_membership.json','episode_order.json','schedule_seed0.npz','schedule_seed1.npz','original_geometry_normalization.json','local_normalization.json']]
save(S/'configs/design_frozen.json',{'utc':now(),'before_training':True,'sha256':{str(p):sha(p)for p in files}})
for p in files+[S/'configs/design_frozen.json']:p.chmod(0o444)
print('FINAL_DESIGN_FROZEN',checks,flush=True)
