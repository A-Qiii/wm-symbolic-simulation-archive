"""RGB/proprio only. No simulator segmentation, depth or camera calibration."""
import torch,numpy as np
from torch.nn import functional as TF
from torchvision.ops import roi_align
from transformers import Dinov2Model
from paths import DINO_WEIGHTS as WEIGHTS
def largest(mask):
 # Components are calculated from RGB thresholds; simulator masks are never read.
 hh,ww=mask.shape;todo=mask.copy().reshape(-1);best=None;bestn=0
 for start in np.flatnonzero(todo):
  if not todo[start]:continue
  todo[start]=False;stack=[int(start)];area=0;x0=ww;x1=0;y0=hh;y1=0
  while stack:
   v=stack.pop();y,x=divmod(v,ww);area+=1;x0=min(x0,x);x1=max(x1,x);y0=min(y0,y);y1=max(y1,y)
   for n in ([v-1]if x else[])+([v+1]if x+1<ww else[])+([v-ww]if y else[])+([v+ww]if y+1<hh else[]):
    if todo[n]:todo[n]=False;stack.append(n)
  if area>bestn:bestn=area;best=(x0,y0,x1+1,y1+1,area)
 return best if bestn>=4 else None
class RGBFront:
 def __init__(self):
  self.net=Dinov2Model.from_pretrained(str(WEIGHTS),local_files_only=True).cuda().eval().requires_grad_(False);self.mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None];self.std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
 def extract(self,paths):
  deterministic=torch.are_deterministic_algorithms_enabled();warn=torch.is_deterministic_algorithms_warn_only_enabled();precision=torch.get_float32_matmul_precision()
  try:
   torch.use_deterministic_algorithms(False);torch.set_float32_matmul_precision('highest')
   with torch.autocast('cuda',enabled=False):return self._extract(paths)
  finally:torch.use_deterministic_algorithms(deterministic,warn_only=warn);torch.set_float32_matmul_precision(precision)
 def _extract(self,paths):
  assert len(paths)==4;images=[];boxes=[];stats=[];robot=None
  for path in paths:
   with np.load(path)as z:
    # Strict key access rather than loading a complete frame dictionary.
    rgb=[z['rgb_static'],z['rgb_gripper']];robot=z['robot_obs'].copy()
   for x in rgb:
    hh,ww=x.shape[:2];top=(224-hh)//2;left=(224-ww)//2;assert min(top,left)>=0;im=torch.from_numpy(x.copy()).permute(2,0,1).float()/255;images.append(TF.pad(im,(left,224-ww-left,top,224-hh-top),value=.5));r,g,b=(x.astype(np.float32)/255).transpose(2,0,1);masks=[(b>.15)&(b>r*1.7)&(b>g*1.5),(r>.22)&(r>g*2)&(r>b*1.7)];bb=[];ss=[]
    for mask in masks:
     q=largest(mask)
     if q:
      x0,y0,x1,y1,area=q;bb.append([max(0,x0+left-2),max(0,y0+top-2),min(224,x1+left+2),min(224,y1+top+2)]);ss.append([x0/ww,y0/hh,x1/ww,y1/hh,(x0+x1)/2/ww,(y0+y1)/2/hh,area/(hh*ww),1.])
     else:bb.append([0,0,1,1]);ss.append([0.]*8)
    boxes.append(bb);stats.append(ss)
  xx=torch.stack(images).cuda()
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):patch=self.net(pixel_values=(xx-self.mean)/self.std).last_hidden_state[:,1:]
  patch=patch.float().transpose(1,2).reshape(8,384,16,16);pool=[]
  for i,bb in enumerate(boxes):pool.append(roi_align(patch[i:i+1],[torch.tensor(bb,device='cuda',dtype=torch.float32)],output_size=(2,2),spatial_scale=1/14,aligned=True).mean((-1,-2)))
  obj=torch.stack(pool).reshape(4,2,2,384).mean(0).permute(1,0,2).reshape(2,768);scene=patch.mean((-1,-2)).reshape(4,2,384).mean(0).reshape(768);ss=np.array(stats,np.float32).reshape(4,2,2,8)[-1].transpose(1,0,2).reshape(2,16);ob=obj.cpu().numpy();sc=scene.cpu().numpy();features=np.concatenate([ob,np.repeat(sc[None],2,axis=0),ss,np.repeat(robot[None],2,axis=0)],-1).astype(np.float32);return features,ss
 def close(self):del self.net;torch.cuda.empty_cache()
