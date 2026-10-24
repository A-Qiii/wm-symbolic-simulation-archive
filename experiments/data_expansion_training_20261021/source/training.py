from common import *
import torch,ast,copy
from models import ObjectAug,Interface
from inputs import load_training as load
from mid_data import batch,crop,warm_prior
from losses import loss_for_batch
for name in ['newmodel','fit']:
 node=next(n for n in ast.parse((M/'source/mid_train.py').read_text()).body if isinstance(n,ast.FunctionDef)and n.name==name);exec(compile(ast.Module(body=[node],type_ignores=[]),'unchanged midstart training '+name,'exec'))
def preflight():
 from mid_data import load as old_load
 ns={'torch':torch,'np':np,'O':O,'OLD':OLD,'S':M,'read':read}
 node=next(n for n in ast.parse((M/'source/mid_data.py').read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='load');exec(compile(ast.Module(body=[node],type_ignores=[]),'old independent normalization reference','exec'),ns)
 a=old_load('train',0,'cpu');b=ns['load']('train',0,'cpu');assert a.keys()==b.keys()and all(torch.equal(a[k],b[k])for k in a)
 initial=[]
 for seed in [0,1]:
  m=newmodel(seed);h=statehash(m);reference=read(M/f'results/fit_cold_mix_seed{seed}.json');assert h==reference['initial_sha256'];initial.append(h)
 data=load('train',0,'cpu');assert data['objects'].shape[0]==868
 weights=[torch.tensor(x)for x in read(CELL/'configs/normalization.json')['class_weights']]
 q=crop(batch(data,[0,120,228,867]),[1,3,0,5]);m=newmodel(0);loss,terms=loss_for_batch(m,q,weights);assert torch.isfinite(loss);loss.backward();grad={name:float(sum(p.grad.abs().sum()for p in mod.parameters()if p.grad is not None))for name,mod in [('object_prior',m.object_prior),('object_observe',m.object_observe),('context_prior',m.context_prior),('context_observe',m.context_observe)]};assert all(v>0 for v in grad.values())
 save(S/'results/training_preflight.json',{'utc':now(),'passed':True,'old_normalized_input_all_tensors_equal':True,'matching_initial_state_hashes':initial,'cropped_loss_finite':True,'gradient_sums':grad,'optimizer_updates':0})
 print('TRAINING_PREFLIGHT_PASS',flush=True)
def phase():
 preflight();save(S/'results/training_started.json',{'utc':now(),'epoch':time.time()})
 for seed in [0,1]:
  done=S/f'results/fit_new868_seed{seed}.json'
  if done.exists():assert read(done)['updates']==1500;continue
  fit('new868',seed)
 paths=[M/f'models/cold_mix_seed{seed}_step1500.pt'for seed in [0,1]]+[S/f'models/new868_seed{seed}_step1500.pt'for seed in [0,1]]
 save(S/'configs/models_frozen.json',{'utc':now(),'sha256':{str(p):sha(p)for p in paths},'new_fits':2,'reused_old_fits':2,'steps_each':1500,'no_selection':True})
 print('FOUR_MODELS_FROZEN',flush=True)
