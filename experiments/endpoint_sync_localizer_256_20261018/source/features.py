from common import *
from net import tensor_hash
import torch,torch.nn.functional as F,fcntl,sys,time
from transformers import Dinov2Model
def main():
 assert read(E/'results/data_gate.json')['passed'];assert not(E/'results/features.json').exists();rows=read(E/'data/public/images.json');dest=E/'data/features';dest.mkdir();start=time.perf_counter()
 def guard(event,args):
  if event=='open'and isinstance(args[0],(str,bytes)):
   p=Path(args[0]).resolve()
   if p.is_relative_to(E/'data/annotation_private'):raise RuntimeError('Encoder cannot read GT')
   if p.is_relative_to(P/'experiments')and not p.is_relative_to(E):raise RuntimeError('No prior experiments in encoder')
 sys.addaudithook(guard);lock=(P/'runs/gpu.lock').open('r+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);torch.cuda.reset_peak_memory_stats();model=Dinov2Model.from_pretrained(DINO,local_files_only=True).cuda().eval().requires_grad_(False);before=tensor_hash(model.state_dict());mu=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None];sd=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None];summary=[]
 for split in ['train','dev','final']:
  selected=[r for r in rows if r['split']==split];out=[]
  for i in range(0,len(selected),8):
   images=[]
   for r in selected[i:i+8]:
    z=np.load(r['path'],allow_pickle=False);assert set(z.files)=={'rgb_static','rgb_gripper'};rgb=z[r['key']];assert hashlib.sha256(rgb.tobytes()).hexdigest()==r['RGB_sha256'];images.append(F.interpolate(torch.from_numpy(rgb).permute(2,0,1)[None].float(),size=(448,448),mode='bilinear',align_corners=False)[0]/255)
   with torch.inference_mode():tokens=model(pixel_values=(torch.stack(images).cuda()-mu)/sd).last_hidden_state[:,1:]
   out.append(tokens.cpu().numpy())
  a=np.concatenate(out);assert a.shape==(len(selected),1024,384);np.savez_compressed(dest/f'{split}.npz',x=a,image_ids=np.array([r['image_id']for r in selected]));summary.append({'split':split,'shape':a.shape,'sha256':sha(dest/f'{split}.npz')});print('FEATURES_COMPLETE',split,flush=True)
 assert tensor_hash(model.state_dict())==before;torch.cuda.synchronize();save(E/'results/features.json',{'utc':now(),'seconds':time.perf_counter()-start,'peak_GPU_bytes':torch.cuda.max_memory_allocated(),'encoder_unchanged':True,'encoder_hash':before,'RGB_only':True,'private_annotation_reads':0,'splits':summary});lock.close()
if __name__=='__main__':main()
