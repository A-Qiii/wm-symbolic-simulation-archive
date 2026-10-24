from common import *
import torch
from models import history
def load(split,seed,device='cuda'):
 data=dict(np.load(O/f'cache/{split}_seed{seed}.npz'));raw=dict(np.load(OLD/f'cache/{split}_geometry_raw.npz'));norm=read(S/'configs/input_normalization.json')['values'][f'geometry_seed{seed}']
 for name,x in raw.items():
  n=norm[name];out=(x-np.array(n['mean'],np.float32))/np.array(n['std'],np.float32);width=8 if name=='objects'else 4
  for j in range(0,x.shape[-1],width):out[...,j:j+width]*=x[...,j+width-1:j+width]>0
  data[name]=np.concatenate([data[name],out.astype(np.float32)],-1)
 return{k:torch.from_numpy(v).to(device)for k,v in data.items()}
def batch(data,ids):return{k:v[ids]for k,v in data.items()}
def crop(b,starts):
 B=len(starts);st=torch.as_tensor(starts,device=b['objects'].device);rows=torch.arange(B,device=st.device)[:,None];out={}
 for k,v in b.items():
  n=v.shape[1];ind=(st[:,None]+torch.arange(n,device=st.device)).clamp_max(n-1);out[k]=v[rows,ind]
 out['step_valid']=out['step_valid']*((torch.arange(6,device=st.device)[None]+st[:,None])<b['step_valid'].sum(1)[:,None])
 return out
def warm_prior(m,b,starts):
 # Only public object/context/availability and requested actions are referenced.
 prior=b['objects'].new_zeros(len(starts),4,128)
 with torch.no_grad():
  for start in sorted(set(int(x)for x in starts)):
   if start==0:continue
   ids=[j for j,x in enumerate(starts)if x==start];q=batch(b,ids);state=m.initial(q['objects'][:,0],q['context'][:,0],q['availability'][:,0])
   for t in range(start):
    state,_=m.imagine(state,q['verb'][:,t],q['pointer'][:,t],q['budget'][:,t])
    if t+1<start:state=m.observe(state,q['objects'][:,t+1],q['context'][:,t+1],q['availability'][:,t+1])
   prior[ids]=state
 return prior.detach()
def rollout(m,b,initial_prior=None):
 if initial_prior is None:return history(m,b['objects'],b['context'],b['availability'],b['verb'],b['pointer'],b['budget'])
 state=m.observe(initial_prior,b['objects'][:,0],b['context'][:,0],b['availability'][:,0]);posts=[state];priors=[]
 for t in range(b['verb'].shape[1]):
  p,_=m.imagine(state,b['verb'][:,t],b['pointer'][:,t],b['budget'][:,t]);state=m.observe(p,b['objects'][:,t+1],b['context'][:,t+1],b['availability'][:,t+1]);posts.append(state);priors.append(p)
 return posts,priors
