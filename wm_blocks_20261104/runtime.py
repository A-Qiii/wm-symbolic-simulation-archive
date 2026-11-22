from common import *
from models import RGBState,HeldHead,ObjectDynamics,pack
import torch,numpy as np
from train import labels_held
def action(a,batch=1):return [torch.full((batch,),v,device='cuda',dtype=torch.long)for v in [int(a[0]=='Place'),NAMES.index(a[1]),{None:0,'A':1,'B':2,'large_top':3}[a[2]]]]
class Runtime:
 def __init__(self,kind,seed,arm='heldout',model_root=None):
  root=Path(model_root)if model_root is not None else ROOT
  self.kind=kind;self.seed=seed;self.arm=arm;self.norm=np.load(root/'models/normalization.npz');self.est=RGBState().cuda();self.head=HeldHead().cuda();self.bce=HeldHead().cuda();self.net=ObjectDynamics(kind).cuda();rec=read(root/f'models/{arm}_{kind}_seed{seed}.json');assert sha(rec['checkpoint'])==rec['sha256']
  assert sha(root/f'models/state_seed{seed}.pt')==rec['state_sha256']and sha(root/f'models/held_ce_seed{seed}.pt')==rec['held_sha256']
  for m,p in [(self.est,root/f'models/state_seed{seed}.pt'),(self.head,root/f'models/held_ce_seed{seed}.pt'),(self.bce,root/f'models/held_bce_seed{seed}.pt'),(self.net,rec['checkpoint'])]:m.load_state_dict(torch.load(p,weights_only=False,map_location='cuda')['model']);m.eval().requires_grad_(False)
 @torch.no_grad()
 def observe(self,f):
  x=torch.as_tensor((f-self.norm['mean'])/self.norm['std'],device='cuda',dtype=torch.float32)
  if x.ndim==2:x=x[None]
  h,p,s,z=self.est(x);return pack(p,s,self.head.probabilities(h))
 @torch.no_grad()
 def predict(self,s,a):return self.net(s,*action(a,len(s)),self.est.sd)
 @torch.no_grad()
 def symbols(self,s):
  assert len(s)==1;return dict(region=self.est.read_regions(s[...,:3],s[...,3:6]).argmax(-1)[0].tolist(),held=labels_held(s[...,6])[0].tolist(),xyz=s[0,:,:3].tolist(),held_probability=s[0,:,6].tolist())
