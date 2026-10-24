from common import *
import torch
from torch import nn
import torch.nn.functional as F
def make():
 m=nn.Sequential(nn.Linear(384,64),nn.SiLU(),nn.Linear(64,15))
 with torch.no_grad():m[-1].bias.reshape(3,5)[:,0].fill_(-2.197224577)
 return m
def unpack(raw):
 z=raw.reshape(*raw.shape[:-1],3,5);return z[...,0],torch.cat([.5*z[...,1:3].tanh(),z[...,3:5].sigmoid()],-1)
def toboxes(g,patch):
 xy=torch.stack([patch%GRID+.5,patch//GRID+.5],-1).to(g.dtype);center=(xy+g[...,:2])/GRID;return torch.cat([center-g[...,2:]/2,center+g[...,2:]/2],-1)
def giou(a,b):
 inter=(torch.minimum(a[...,2:],b[...,2:])-torch.maximum(a[...,:2],b[...,:2])).clamp(min=0).prod(-1);ar=(a[...,2:]-a[...,:2]).clamp(min=0).prod(-1);br=(b[...,2:]-b[...,:2]).clamp(min=0).prod(-1);u=ar+br-inter;c=(torch.maximum(a[...,2:],b[...,2:])-torch.minimum(a[...,:2],b[...,:2])).clamp(min=0).prod(-1);return inter/u.clamp(min=1e-9)-(c-u)/c.clamp(min=1e-9)
def loss(raw,y,v,b):
 logits,g=unpack(raw);pr=logits.sigmoid();bce=F.binary_cross_entropy_with_logits(logits,y,reduction='none');weight=torch.where(torch.arange(3,device=y.device)[None,None,:]==0,torch.where(y>0,.25,.75),torch.where(y>0,.25*(1-pr).square(),.75*pr.square()));focal=((bce*weight*v[:,None,:]).sum((1,2))/y.sum((1,2)).clamp(min=1)).mean();ii,pp,ss=torch.where(y>0)
 if len(ii):reg=F.smooth_l1_loss(g[ii,pp,ss],b[ii,pp,ss]);box=(1-giou(toboxes(g[ii,pp,ss],pp),toboxes(b[ii,pp,ss],pp))).mean()
 else:reg=raw.sum()*0;box=reg
 return focal+5*reg+2*box,torch.stack([focal.detach(),reg.detach(),box.detach()])
def iou(a,b):
 a=np.asarray(a);b=np.asarray(b);inter=np.maximum(np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]),0).prod();union=np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-inter;return float(inter/max(float(union),1e-9))
def decode(raw):
 with torch.inference_mode():log,g=unpack(torch.tensor(raw));scores=log.sigmoid().numpy();boxes=toboxes(g,torch.arange(GRID*GRID)[:,None].expand(-1,3)).clamp(0,1).numpy()
 out=[]
 for cat,slots in [('drawer_front',[0]),('gripper_finger',[1,2])]:
  candidates=[(int(p),s)for s in slots for p in np.where(scores[:,s]>=.5)[0]];candidates.sort(key=lambda a:(-float(scores[a]),a[0],a[1]));keep=[]
  for p,s in candidates:
   b=boxes[p,s]
   if b[2]<=b[0]or b[3]<=b[1]or any(iou(b,q['box'])>.5 for q in keep):continue
   keep.append({'category':cat,'score':float(scores[p,s]),'box':b.tolist(),'patch':p,'slot':s})
  out+=keep
 return out
def tensor_hash(state):return hashlib.sha256(b''.join(k.encode()+v.detach().cpu().numpy().tobytes()for k,v in sorted(state.items()))).hexdigest()
