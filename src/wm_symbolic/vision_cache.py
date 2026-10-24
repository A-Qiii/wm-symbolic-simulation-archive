"""Public RGB/mask geometry only; frozen DINO patches, visible ROI tokens."""
from pathlib import Path
import json,fcntl,os,hashlib
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import roi_align
from transformers import Dinov2Model

WEIGHTS=Path('/home/robot/research/models/facebook--dinov2-small/ed25f3a31f01632728cabb09d1542f84ab7b0056')
PUBLIC_KEYS={'rgb_static','rgb_gripper','depth_static','depth_gripper','robot_obs','seg_static','seg_gripper'}

class VisibleRegistry:
 """Register masks from pixels. No semantic ID or pose from the simulator."""
 def __init__(self):self.ids=[];self.colour_ids={};self.max_overflow=0
 @staticmethod
 def grouped(mask):
  mask=mask.copy();mask[(mask>=0)&((mask&((1<<24)-1))==0)]=0
  return mask
 def update(self,frames):
  area={}
  for frame in frames:
   for cam in ['static','gripper']:
    mask=self.grouped(frame['seg_'+cam]);rgb=frame['rgb_'+cam].astype(float)/255
    for uid in np.unique(mask):
     uid=int(uid)
     if uid<0:continue
     pixels=mask==uid;area[uid]=area.get(uid,0)+int(pixels.sum());mean=rgb[pixels].mean(0)
     # Colour is a visible diagnostic registration cue; never hidden metadata.
     colour=0 if mean[0]>.35 and mean[0]>mean[1]*1.7 and mean[0]>mean[2]*1.7 else (1 if mean[2]>.35 and mean[2]>mean[0]*1.7 and mean[2]>mean[1]*1.7 else (2 if mean[0]>.35 and mean[2]>.35 and mean[1]<min(mean[0],mean[2])*.6 else None))
     if colour is not None and colour not in self.colour_ids:self.colour_ids[colour]=uid
  desired=list(dict.fromkeys(list(self.colour_ids.values())+self.ids+sorted(area,key=area.get,reverse=True)))
  self.max_overflow=max(self.max_overflow,max(0,len(desired)-12));self.ids=desired[:12]
  return self.ids

class FrozenVision:
 def __init__(self,project,device='cuda:0',reserve_gpu=True):
  self.project=Path(project);self.device=device
  self.lock=None
  if reserve_gpu:
   self.lock=(self.project/'runs/gpu.lock').open('a+');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB);self.lock.seek(0);self.lock.truncate();self.lock.write(str(os.getpid()));self.lock.flush()
  self.model=Dinov2Model.from_pretrained(str(WEIGHTS),local_files_only=True).eval().requires_grad_(False).to(device)
  self.mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None];self.std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
 def encode(self,paths,registry):
  frames=[]
  for path in paths:
   with np.load(path) as z:
    forbidden=[k for k in z.files if k not in PUBLIC_KEYS and not k.startswith('calib_')]
    if forbidden:raise ValueError('Hidden or undeclared inputs: '+str(forbidden))
    frames.append({k:z[k] for k in z.files})
  assert len(frames)==4
  ids=registry.update(frames);images=[];geometries=[];masks=[]
  for frame in frames:
   for cam in ['static','gripper']:
    rgb=frame['rgb_'+cam];h,w=rgb.shape[:2];top=(224-h)//2;left=(224-w)//2
    if top<0 or left<0:raise ValueError('Image larger than frozen 224 input protocol')
    image=torch.from_numpy(rgb.copy()).permute(2,0,1).float()/255
    images.append(F.pad(image,(left,224-w-left,top,224-h-top),value=.5))
    seg=registry.grouped(frame['seg_'+cam]);boxes=[];roi_masks=[];geom=[]
    for slot in range(12):
     mask=seg==ids[slot] if slot<len(ids) else np.zeros(seg.shape,bool)
     y,x=np.where(mask);visible=len(y)>=2
     x0,x1=(int(x.min()+left),int(x.max()+left+1)) if visible else (0,1)
     y0,y1=(int(y.min()+top),int(y.max()+top+1)) if visible else (0,1)
     boxes.append([x0,y0,x1,y1]);padded=np.pad(mask.astype(np.float32),((top,224-h-top),(left,224-w-left)))
     roi_masks.append(padded);geom.append([x0/224,y0/224,x1/224,y1/224,float(visible)])
    geometries.append((boxes,geom));masks.append(roi_masks)
  x=torch.stack(images).to(self.device)
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   patch=self.model(pixel_values=(x-self.mean)/self.std).last_hidden_state[:,1:]
  assert patch.shape==(8,256,384)
  patch=patch.float().transpose(1,2).reshape(8,384,16,16)
  tokens=[];geometry=[];valid=[];global_targets=[]
  for i,(boxes,geom) in enumerate(geometries):
   boxes=torch.tensor(boxes,device=self.device,dtype=torch.float32)
   rois=roi_align(patch[i:i+1],[boxes],output_size=(2,2),spatial_scale=1/14,aligned=True)
   masks_tensor=torch.tensor(np.stack(masks[i]),device=self.device)[:,None]
   mask_rois=roi_align(masks_tensor,torch.cat([torch.arange(12,device=self.device)[:,None],boxes],1),output_size=(2,2),aligned=True)[:,0]
   local=rois.permute(0,2,3,1).reshape(48,384)
   # Mask occupancy remains an explicit token attribute; not an invisible contour.
   local_geometry=[]
   for j,g in enumerate(geom):
    for k in range(4):local_geometry.append(g+[float(mask_rois[j,k//2,k%2]),i%2,(i//2)/3])
   tokens.append(local);geometry.extend(local_geometry);valid.extend([bool(g[4]) for g in geom for _ in range(4)])
   coarse=F.adaptive_avg_pool2d(patch[i:i+1],(4,4)).flatten(2).transpose(1,2)[0];tokens.append(coarse)
   geometry.extend([[x/4,y/4,(x+1)/4,(y+1)/4,1.,1.,i%2,(i//2)/3] for y in range(4) for x in range(4)]);valid.extend([True]*16)
   global_targets.append(coarse)
  target=torch.stack(global_targets).reshape(4,2,16,384).mean(0).reshape(-1)
  return {'tokens':torch.cat(tokens).cpu().half().numpy(),'geometry':np.asarray(geometry,np.float32),'valid':np.asarray(valid,bool),
   'robot':frames[-1]['robot_obs'].astype(np.float32),'reconstruction':target.cpu().half().numpy(),
   'visible_registry_ids':np.array(ids,np.int64),'overflow_count':np.array(registry.max_overflow)}

def cache_manifest(project):
 from wm_symbolic.checkpointing import PauseRequest
 pause=PauseRequest()
 p=Path(project);m=json.loads((p/'data/data_manifest.json').read_text());encoder=FrozenVision(p);episodes={}
 for r in m['records']:episodes.setdefault(r['episode_id'],[]).append(r)
 cache=p/'data/features/dinov2_v1';cache.mkdir(parents=True,exist_ok=True);items=[]
 for eid,rows in episodes.items():
  rows.sort(key=lambda r:r['step']);ed=Path(rows[0]['obs_before']).parent.parent;reg=VisibleRegistry()
  boundaries=[[ed/f'root_observation_{j}.npz' for j in range(4)]]+[[Path(r['obs_after']).parent/f'boundary_history_{j}.npz' for j in range(4)] for r in rows]
  for step,paths in enumerate(boundaries):
   out=cache/f'{eid}_{step:02d}.npz'
   # Replay the visible registration even if a previous cache file exists.
   if out.exists():
    frames=[dict(np.load(path)) for path in paths];reg.update(frames)
   else:
    temporary=out.with_suffix('.tmp.npz');np.savez(temporary,**encoder.encode(paths,reg));temporary.replace(out)
   items.append({'episode_id':eid,'root_id':rows[0]['root_id'],'split':rows[0]['split'],'step':step,'path':str(out),'public_inputs':[str(x) for x in paths]})
  if pause.requested:
   (p/'runs/PAUSED_BY_USER.json').write_text(json.dumps({'status':'paused','stage':'feature_cache','last_complete_episode':eid,'cached_items_in_scan':len(items),'signal':pause.reason,'resume_requires_user_authorization':True},indent=2));print('CACHE_PAUSED_SAFELY',eid,flush=True);raise SystemExit(0)
  print('CACHED',eid,len(rows)+1,flush=True)
 index={'items':items,'backbone':str(WEIGHTS),'frozen':True,'image_protocol':'Centre-pad native RGB with 0.5 to 224; ImageNet mean/std; no horizontal flip/hue change',
  'roi':'2x2 ROIAlign over 16x16x384 patches, masks preserved; 12 slots plus 4x4 coarse grid per view and frame',
  'geometry':'Visible pixel boxes, occupancy, view and time; RGB-only model, depth retained for later declared comparison',
  'only_public_inputs':True,'checkpoint_sha256':hashlib.sha256((WEIGHTS/'model.safetensors').read_bytes()).hexdigest()}
 (cache/'index.json').write_text(json.dumps(index,indent=2));print('CACHE_COMPLETE',len(items),torch.cuda.max_memory_allocated(),flush=True)
