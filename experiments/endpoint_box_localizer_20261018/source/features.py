from common import *
import torch,torch.nn.functional as F,fcntl,sys,time,shutil
from transformers import Dinov2Model
from net import tensor_hash

def main():
 assert read(E/'results/annotation_quality_gate.json')['passed'];dest=E/'data/features/tokens.npz';assert not dest.exists();dest.parent.mkdir()
 for p,h in read(E/'configs/pre_annotation_frozen.json')['sha256'].items():assert sha(p)==h,p
 for p,h in read(E/'configs/annotations_frozen.json')['sha256'].items():assert sha(p)==h,p
 rows=read(E/'data/public/images.json');save(E/'configs/feature_code_frozen.json',{'utc':now(),'sha256':{str(Path(__file__).resolve()):sha(__file__),str(DINO/'model.safetensors'):sha(DINO/'model.safetensors')},'annotation_read_during_encoding':False})
 def guard(event,args):
  if event=='open'and isinstance(args[0],(str,bytes)):
   p=Path(args[0]).resolve()
   if p.is_relative_to(P/'experiments')and not p.is_relative_to(E):raise RuntimeError('No other experiments in visual extraction')
   if p.is_relative_to(E/'data/annotations'):raise RuntimeError('RGB encoder cannot read labels')
 sys.addaudithook(guard);lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);torch.cuda.init();torch.cuda.reset_peak_memory_stats();start=time.perf_counter();model=Dinov2Model.from_pretrained(DINO,local_files_only=True).cuda().eval().requires_grad_(False);before=tensor_hash(model.state_dict());out=[];mu=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None];sd=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
 for i in range(0,len(rows),8):
  ims=[]
  for r in rows[i:i+8]:
   z=np.load(r['path']);assert set(z.files)=={'rgb_static','rgb_gripper'};rgb=z[r['key']];ims.append(F.interpolate(torch.from_numpy(rgb).permute(2,0,1)[None].float(),size=(224,224),mode='bilinear',align_corners=False)[0]/255)
  with torch.inference_mode():tok=model(pixel_values=(torch.stack(ims).cuda()-mu)/sd).last_hidden_state[:,1:]
  out.append(tok.cpu().numpy())
 assert tensor_hash(model.state_dict())==before;torch.cuda.synchronize();np.savez_compressed(dest,tokens=np.concatenate(out),image_ids=np.array([r['image_id']for r in rows]));save(E/'results/features.json',{'utc':now(),'images':40,'shape':[40,256,384],'encoder_unchanged':True,'encoder_hash':before,'RGB_only':True,'annotation_reads_during_encoding':0,'seconds':time.perf_counter()-start,'peak_bytes':torch.cuda.max_memory_allocated()});shutil.copy2(__file__,E/'source/features.py');print('FROZEN_DINO_BOX_FEATURES_READY',flush=True)
if __name__=='__main__':main()
