from common import *
import torch
from mid_data import load as load_old
def new_data(split,seed,device='cuda'):
 data=dict(np.load(S/f'cache/{split}_seed{seed}.npz'));raw=dict(np.load(S/f'cache/{split}_geometry_raw.npz'));norm=read(S/'configs/input_normalization.json')['values'][f'geometry_seed{seed}']
 for name,x in raw.items():
  n=norm[name];out=(x-np.array(n['mean'],np.float32))/np.array(n['std'],np.float32);width=8 if name=='objects'else 4
  for j in range(0,x.shape[-1],width):out[...,j:j+width]*=x[...,j+width-1:j+width]>0
  data[name]=np.concatenate([data[name],out.astype(np.float32)],-1)
 return{k:torch.from_numpy(v).to(device)for k,v in data.items()}
def load_training(split,seed,device='cuda'):
 assert split=='train';a=load_old('train',seed,device);b=new_data('train',seed,device);assert a.keys()==b.keys()
 return{k:torch.cat([a[k],b[k]],0)for k in a}
def cohort(name,seed,device='cuda'):
 if name=='old_development':return load_old('development',seed,device),read(O/'cache/development_metadata.json')
 return new_data(name,seed,device),read(S/f'cache/{name}_metadata.json')
