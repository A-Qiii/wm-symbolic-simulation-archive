"""Generic frozen DINO, CURRENT RGB only. No semantic model imports."""
from common import *
from registry import extract
import torch,torch.nn.functional as F,fcntl,time
from torchvision.ops import roi_align
from transformers import Dinov2Model
class Encoder:
 def __init__(self,device='cuda'):
  self.device=device;self.net=Dinov2Model.from_pretrained(DINO,local_files_only=True).eval().requires_grad_(False).to(device)
  self.mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None];self.std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
 def encode(self,paths):
  images=[];geometry=[];robots=[]
  for path in paths:
   with np.load(path)as z:
    assert set(z.files)=={'rgb_static','rgb_gripper','robot_obs'}
    frames={k:z[k].copy()for k in z.files}
   robots.append(frames['robot_obs'])
   for cam in ['static','gripper']:
    rgb=frames['rgb_'+cam];h,w=rgb.shape[:2];top,left=(224-h)//2,(224-w)//2;assert top>=0 and left>=0
    images.append(F.pad(torch.from_numpy(rgb).permute(2,0,1).float()/255,(left,224-w-left,top,224-h-top),value=.5));masks,_=extract(rgb);boxes=[];gg=[];pm=[]
    for j in range(3):
     yy,xx=np.where(masks[j]);vis=len(yy)>=4;x0,x1=(int(xx.min()+left),int(xx.max()+left+1))if vis else(0,1);y0,y1=(int(yy.min()+top),int(yy.max()+top+1))if vis else(0,1)
     boxes.append([x0,y0,x1,y1]);gg.append([x0/224,y0/224,x1/224,y1/224,float(vis)]);pm.append(np.pad(masks[j].astype(np.float32),((top,224-h-top),(left,224-w-left))))
    geometry.append((boxes,gg,pm))
  x=torch.stack(images).to(self.device)
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):patch=self.net(pixel_values=(x-self.mean)/self.std).last_hidden_state[:,1:]
  patch=patch.float().transpose(1,2).reshape(len(images),384,16,16);local=[];geom=[];scene=[]
  for i,(boxes,g,pm)in enumerate(geometry):
   boxes=torch.tensor(boxes,dtype=torch.float32,device=self.device);roi=roi_align(patch[i:i+1],[boxes],output_size=(2,2),spatial_scale=1/14,aligned=True).permute(0,2,3,1).reshape(3,4,384)
   masks=torch.tensor(np.stack(pm),device=self.device)[:,None];occ=roi_align(masks,torch.cat([torch.arange(3,device=self.device)[:,None],boxes],1),output_size=(2,2),aligned=True)[:,0].reshape(3,4)
   local.append(roi.cpu().half().numpy());geom.append(np.array([[row+[float(occ[j,k]),i%2,1.]for k in range(4)]for j,row in enumerate(g)],np.float32));scene.append(F.adaptive_avg_pool2d(patch[i:i+1],(4,4)).flatten(2).transpose(1,2)[0].cpu().half().numpy())
  return [{'local_tokens':np.stack(local[2*i:2*i+2]),'geometry':np.stack(geom[2*i:2*i+2]),'global_grid':np.stack(scene[2*i:2*i+2]),'robot_history':np.asarray(robots[i],np.float32)}for i in range(len(paths))]
def main():
 lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);rows=read(E/'data/public/states.json');missing=[r for r in rows if not(E/'data/public/features'/f"{r['id']}.npz").exists()];enc=Encoder();start=time.monotonic()
 for k in range(0,len(missing),8):
  batch=missing[k:k+8];fs=enc.encode([r['path']for r in batch])
  for r,f in zip(batch,fs):np.savez_compressed(E/'data/public/features'/f"{r['id']}.npz",**f)
  if k%80==0:print('CACHED',k+len(batch),'/',len(missing),'seconds',round(time.monotonic()-start,1),flush=True)
  if k>=48 and not(E/'results/first50_storage.json').exists():
   ps=[E/'data/public/features'/f"{r['id']}.npz"for r in missing[:50]];save(E/'results/first50_storage.json',{'states':50,'feature_bytes':sum(p.stat().st_size for p in ps),'observation_bytes':sum(Path(r['path']).stat().st_size for r in missing[:50]),'disk_free_bytes':__import__('shutil').disk_usage(E).free})
 save(E/'results/feature_cache.json',{'newly_encoded':len(missing),'reused_generic_raw_cache':len(rows)-len(missing),'all_states':len(rows),'DINO_sha256':sha(DINO/'model.safetensors'),'current_frame_only':True,'learned_semantic_weights_loaded':False,'elapsed_seconds':time.monotonic()-start});print('FEATURES_COMPLETE',flush=True)
if __name__=='__main__':main()
