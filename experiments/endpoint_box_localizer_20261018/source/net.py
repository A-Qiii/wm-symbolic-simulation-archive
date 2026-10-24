from common import *
import torch
from torch import nn
import torch.nn.functional as F

def make():
 net=nn.Sequential(nn.Linear(384,64),nn.SiLU(),nn.Linear(64,10))
 with torch.no_grad():net[-1].bias[:2].fill_(-2.197224577)
 return net

def unpack(raw):
 logits=raw[...,:2];g=raw[...,2:].reshape(*raw.shape[:-1],2,4)
 return logits,torch.cat([.5*g[...,:2].tanh(),g[...,2:].sigmoid()],-1)

def toboxes(g,patch):
 xx=patch%16;yy=patch//16;c=torch.stack([xx+.5,yy+.5],-1).to(g.dtype)/16+g[...,:2]/16;wh=g[...,2:]
 return torch.cat([c-wh/2,c+wh/2],-1)

def giou(a,b):
 inter=(torch.minimum(a[...,2:],b[...,2:])-torch.maximum(a[...,:2],b[...,:2])).clamp(min=0).prod(-1);area_a=(a[...,2:]-a[...,:2]).clamp(min=0).prod(-1);area_b=(b[...,2:]-b[...,:2]).clamp(min=0).prod(-1);union=area_a+area_b-inter;enc=(torch.maximum(a[...,2:],b[...,2:])-torch.minimum(a[...,:2],b[...,:2])).clamp(min=0).prod(-1)
 return inter/union.clamp(min=1e-9)-(enc-union)/enc.clamp(min=1e-9)

def loss(raw,y,valid,target):
 logits,g=unpack(raw);p=logits.sigmoid();bce=F.binary_cross_entropy_with_logits(logits,y,reduction='none');weight=torch.where(y>0,.25*(1-p).square(),.75*p.square());focal=((bce*weight*valid[:,None,:]).sum((1,2))/y.sum((1,2)).clamp(min=1)).mean();ii,pp,cc=torch.where(y>0)
 if len(ii):
  reg=F.smooth_l1_loss(g[ii,pp,cc],target[ii,pp,cc]);box=1-giou(toboxes(g[ii,pp,cc],pp),toboxes(target[ii,pp,cc],pp)).mean()
 else:reg=raw.sum()*0;box=raw.sum()*0
 return focal+5*reg+2*box,torch.stack([focal.detach(),reg.detach(),box.detach()])

def iou(a,b):
 a=np.asarray(a);b=np.asarray(b);inter=np.maximum(np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]),0).prod();u=np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-inter;return float(inter/max(float(u),1e-9))

def decode(raw):
 with torch.inference_mode():
  log,g=unpack(torch.tensor(raw));s=log.sigmoid();ss=s.T.reshape(1,2,16,16);peaks=ss==F.max_pool2d(ss,3,1,1);mask=peaks.reshape(2,256).T&(s>=.5);allboxes=toboxes(g,torch.arange(256)[:,None].expand(256,2)).clamp(0,1).numpy();s=s.numpy();mask=mask.numpy()
 out=[]
 for c in range(2):
  indices=sorted(np.where(mask[:,c])[0],key=lambda k:(-s[k,c],int(k)));selected=[]
  for k in indices:
   b=allboxes[k,c]
   if b[2]<=b[0]or b[3]<=b[1]or any(iou(b,q['box'])>.3 for q in selected):continue
   selected.append({'category':CATS[c],'score':float(s[k,c]),'box':b.tolist(),'patch':int(k)})
   if len(selected)>=(1 if c==0 else 2):break
  out.extend(selected)
 return out

def tensor_hash(state):return hashlib.sha256(b''.join(k.encode()+v.detach().cpu().numpy().tobytes()for k,v in sorted(state.items()))).hexdigest()
