from common import *
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from models import tensor_hash
import torch,torch.nn.functional as F,time,fcntl,sys
from transformers import Dinov2Model
def geometry(b,g,w,h):
 b=np.array(b,float);g=np.array(g,float);bw=b[2:]-b[:2];gw=g[2:]-g[:2];bc=(b[2:]+b[:2])/2;gc=(g[2:]+g[:2])/2;over=np.maximum(0,np.minimum(b[2:],g[2:])-np.maximum(b[:2],g[:2]));inter=float(np.prod(over));iou=inter/(float(np.prod(bw)+np.prod(gw))-inter);return np.r_[(bc-gc)/[w,h],np.log(bw/gw),iou].astype(np.float32)
def main():
 assert read(E/'results/data_gate.json')['passed'];assert not(E/'results/features.json').exists();manifest=read(E/'data/model_inputs.json');jobs=[];lookup={};pairs=[]
 for r in manifest:
  ids=[]
  for key in ['object_bbox','gripper_bbox']:
   ident=(r['path'],r['key'],tuple(r[key]))
   if ident not in lookup:lookup[ident]=len(jobs);jobs.append({'path':r['path'],'key':r['key'],'bbox':r[key],'RGB_sha256':r['RGB_sha256']})
   ids.append(lookup[ident])
  pairs.append(ids)
 def guard(event,args):
  if event=='open'and isinstance(args[0],(str,bytes)):
   p=Path(os.fsdecode(args[0])).resolve()
   if p.is_relative_to(P/'experiments')and not p.is_relative_to(E):raise RuntimeError('No old experiment read in encoder')
   if p.is_relative_to(E/'data/private')or p.is_relative_to(E/'data/supervision')or p==E/'data/records.json':raise RuntimeError('Hidden state or labels forbidden to encoder')
 sys.addaudithook(guard);lock=(P/'runs/gpu.lock').open('r+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.reset_peak_memory_stats();start=time.perf_counter();model=Dinov2Model.from_pretrained(DINO,local_files_only=True).cuda().eval().requires_grad_(False);before=tensor_hash(model.state_dict());assert before=='3a697b872089933d345fd71087af86e85ff31886cf1e2654763dc1884ef2af27';mu=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None];sd=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None];out=[];crop_hashes=[]
 for i in range(0,len(jobs),32):
  batch=[]
  for job in jobs[i:i+32]:
   with np.load(job['path'],allow_pickle=False)as z:
    assert set(z.files)=={'rgb_static','rgb_gripper'};rgb=z[job['key']];assert hashlib.sha256(rgb.tobytes()).hexdigest()==job['RGB_sha256'];x1,y1,x2,y2=job['bbox'];crop=rgb[y1:y2,x1:x2].copy();assert crop.size;crop_hashes.append(hashlib.sha256(str(crop.shape).encode()+crop.tobytes()).hexdigest());batch.append(F.interpolate(torch.from_numpy(crop).permute(2,0,1)[None].float(),size=(224,224),mode='bilinear',align_corners=False)[0]/255.)
  with torch.inference_mode():f=model(pixel_values=(torch.stack(batch).cuda()-mu)/sd).last_hidden_state[:,1:].mean(1)
  out.append(f.cpu().numpy())
  if i%256==0:print('CROPS_ENCODED',min(i+32,len(jobs)),len(jobs),flush=True)
 features=np.concatenate(out);assert features.shape==(len(jobs),384);projection=np.load(E/'configs/visual_projection.npy');projected=features@projection;assert tensor_hash(model.state_dict())==before;summary=[]
 for split in ['train','dev']:
  selected=[i for i,r in enumerate(manifest)if r['split']==split];obj=np.stack([projected[pairs[i][0]]for i in selected]);grip=np.stack([projected[pairs[i][1]]for i in selected]);geo=np.stack([geometry(manifest[i]['object_bbox'],manifest[i]['gripper_bbox'],manifest[i]['width'],manifest[i]['height'])for i in selected]);pair=np.concatenate([obj,grip,geo],axis=1).astype(np.float32);dest=E/f'data/features/{split}.npz';save_raw_npz(dest,x_object=obj.astype(np.float32),x_pair=pair,x_geometry=geo,record_indices=np.array([manifest[i]['record_index']for i in selected]),block_crop_hash=np.array([crop_hashes[pairs[i][0]]for i in selected]),gripper_crop_hash=np.array([crop_hashes[pairs[i][1]]for i in selected]));summary.append({'split':split,'rows':len(selected),'sha256':sha(dest)})
 torch.cuda.synchronize();save(E/'results/features.json',{'utc':now(),'seconds':time.perf_counter()-start,'encoder_hash':before,'encoder_unchanged':True,'unique_crops':len(jobs),'query_rows':len(manifest),'peak_GPU_bytes':torch.cuda.max_memory_allocated(),'features':summary,'forbidden_private_label_reads':0,'model_inputs':'RGB crops and relative2D boxes only','crop_side':224,'projection_sha256':sha(E/'configs/visual_projection.npy')});lock.close();print('FEATURES_COMPLETE',flush=True)
if __name__=='__main__':main()
