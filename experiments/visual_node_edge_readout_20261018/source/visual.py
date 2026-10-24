"""Pure two-RGB input; explicit object and camera-role endpoint descriptors."""
from common import *
from registry import extract
import torch
import torch.nn.functional as F
from torchvision.ops import roi_align

def load_rgb(path):
 with np.load(path,allow_pickle=False)as z:
  if set(z.files)!={'rgb_static','rgb_gripper'}:raise ValueError('Pure visual encoder accepts exactly two RGB arrays, no robot/state/label fields')
  out=[z[k].copy()for k in['rgb_static','rgb_gripper']]
 for x in out:assert x.dtype==np.uint8 and x.ndim==3 and x.shape[2]==3
 return out

def object_regions(rgb):
 masks,_=extract(rgb);h,w=rgb.shape[:2];boxes=[];geom=[]
 for mask in masks:
  yy,xx=np.where(mask);vis=len(yy)>=4
  box=[int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1]if vis else[0,0,1,1]
  boxes.append(box);geom.append([box[0]/w,box[1]/h,box[2]/w,box[3]/h,1.]if vis else[0.]*5)
 return np.array(boxes,np.float32),np.array(geom,np.float32)

def relative_geometry(geometry,head):
 cam=ROIS[head]['camera'];a=geometry[:,cam];b=np.array(ROIS[head]['box'],np.float32);size=b[2:]-b[:2]
 center=(a[...,:2]+a[...,2:4])/2;obj_size=np.maximum(a[...,2:4]-a[...,:2],1e-3)
 r=np.concatenate([(center-(b[:2]+b[2:])/2)/size,np.log(obj_size/size),a[...,4:5]],-1)
 return (r*a[...,4:5]).astype(np.float32)

def build_vectors(raw,norm):
 def project(x):return (((x.astype(np.float32)-norm['mean'])/norm['std'])@norm['basis'])/norm['scale']
 g=raw['geometry'];obj=project(raw['objects']).mean(3)*g[...,4:5]
 node=np.concatenate([obj.transpose(0,2,1,3).reshape(len(g),3,64),g.transpose(0,2,1,3).reshape(len(g),3,10)],-1).astype(np.float32)
 result={'node':node}
 for k,head in enumerate(HEADS):
  ep=project(raw['endpoints'][:,k]).reshape(len(g),128)
  result[head]=np.concatenate([node,np.repeat(ep[:,None],3,axis=1),relative_geometry(g,head)],-1).astype(np.float32)
 assert result['node'].shape[1:]==(3,74)
 assert all(result[h].shape[1:]==(3,207)for h in HEADS)
 return result

class Encoder:
 def __init__(self):
  from transformers import Dinov2Model
  self.net=Dinov2Model.from_pretrained(DINO,local_files_only=True).cuda().eval().requires_grad_(False)
  self.mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None];self.std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
 def encode(self,paths):
  images=[];meta=[]
  for path in paths:
   for cam,rgb in enumerate(load_rgb(path)):
    h,w=rgb.shape[:2];top,left=(224-h)//2,(224-w)//2;assert top>=0 and left>=0
    images.append(F.pad(torch.from_numpy(rgb).permute(2,0,1).float()/255,(left,224-w-left,top,224-h-top),value=.5))
    box,geom=object_regions(rgb);box+=np.array([left,top,left,top])
    role='In'if cam==0 else'Holding';b=np.array(ROIS[role]['box'])*np.array([w,h,w,h])+np.array([left,top,left,top]);meta.append((box,geom,b))
  x=torch.stack(images).cuda()
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):patch=self.net(pixel_values=(x-self.mean)/self.std).last_hidden_state[:,1:]
  patch=patch.float().transpose(1,2).reshape(len(images),384,16,16);objects=[];geometry=[];ends=[]
  for i,(boxes,geom,ep)in enumerate(meta):
   boxes=torch.tensor(np.concatenate([boxes,ep[None]],0),dtype=torch.float32,device='cuda')
   roi=roi_align(patch[i:i+1],[boxes],output_size=(2,2),spatial_scale=1/14,aligned=True).permute(0,2,3,1).reshape(4,4,384)
   objects.append(roi[:3].cpu().half().numpy());ends.append(roi[3].cpu().half().numpy());geometry.append(geom)
  return [{'objects':np.stack(objects[2*i:2*i+2]),'geometry':np.stack(geometry[2*i:2*i+2]),'endpoints':np.stack([ends[2*i+1],ends[2*i]])}for i in range(len(paths))]

