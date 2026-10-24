from common import *
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch,torch.nn.functional as F,fcntl,time
from transformers import Dinov2Model
def main():
 assert not(E/'results/features.json').exists();guard('features');jobs=read(E/'data/context_jobs.json');queries=read(E/'data/query_context_map.json');lock=(P/'runs/gpu.lock').open('r+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.reset_peak_memory_stats();start=time.perf_counter();model=Dinov2Model.from_pretrained(DINO,local_files_only=True).cuda().eval().requires_grad_(False);h=tensor_hash(model.state_dict());assert h=='3a697b872089933d345fd71087af86e85ff31886cf1e2654763dc1884ef2af27';mu=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None];sd=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None];out=[]
 for i in range(0,len(jobs),32):
  batch=[]
  for job in jobs[i:i+32]:
   z=np.load(job['path']);assert set(z.files)=={'rgb_static','rgb_gripper'};rgb=z['rgb_static'];assert hashlib.sha256(rgb.tobytes()).hexdigest()==job['RGB_sha256'];batch.append(F.interpolate(torch.from_numpy(rgb.copy()).permute(2,0,1)[None].float(),size=(224,224),mode='bilinear',align_corners=False)[0]/255.)
  with torch.inference_mode():f=model(pixel_values=(torch.stack(batch).cuda()-mu)/sd).last_hidden_state[:,1:].mean(1)
  out.append(f.cpu().numpy())
 projection=np.load(E/'configs/visual_projection.npy');full=(np.concatenate(out)@projection).astype(np.float32);assert full.shape==(264,128);assert tensor_hash(model.state_dict())==h;npz(E/'data/static_context_cache.npz',features=full,paths=np.array([j['path']for j in jobs]),RGB_sha256=np.array([j['RGB_sha256']for j in jobs]));idx={j['path']:i for i,j in enumerate(jobs)};meta={q['record_index']:q for q in queries};summaries=[]
 for split in ['train','dev']:
  base=np.load(B/f'data/features/{split}.npz');context=np.stack([full[idx[meta[int(i)]['path']]]for i in base['record_indices']]);x=np.concatenate([base['x_pair'],context],1).astype(np.float32);assert x.shape==(920 if split=='train' else 470,389);f=E/f'data/{split}_features.npz';npz(f,x=x,record_indices=base['record_indices'],context_indices=np.array([idx[meta[int(i)]['path']]for i in base['record_indices']]));summaries.append({'split':split,'rows':len(x),'sha256':sha(f)})
 torch.cuda.synchronize();save(E/'results/features.json',{'utc':now(),'frames':264,'features':summaries,'encoder_hash_before_after':h,'encoder_unchanged':True,'projection_sha256':sha(E/'configs/visual_projection.npy'),'seconds':time.perf_counter()-start,'peak_GPU_bytes':torch.cuda.max_memory_allocated(),'private_or_label_reads':0,'new_images':0});lock.close();print('CONTEXT_FEATURES_COMPLETE',summaries)
if __name__=='__main__':main()
