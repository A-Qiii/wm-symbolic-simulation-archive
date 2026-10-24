from common import *
from net import tensor_hash
import torch,torch.nn.functional as F,fcntl,sys,time,gc
from transformers import Dinov2Model

def main():
 assert not(E/'results/features.json').exists()
 check_manifest(E/'configs/pre_execution_frozen.json')
 rows=read(E/'data/public/images.json')
 def guard(event,args):
  if event=='open'and isinstance(args[0],(str,bytes)):
   p=Path(args[0]).resolve()
   if p.is_relative_to(P/'experiments')and not p.is_relative_to(E):raise RuntimeError('No other experiments during RGB encoding')
   if p.is_relative_to(E/'data/annotations'):raise RuntimeError('Encoder cannot read annotations')
 sys.addaudithook(guard)
 lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 torch.set_num_threads(4);torch.cuda.init()
 model=Dinov2Model.from_pretrained(DINO,local_files_only=True).cuda().eval().requires_grad_(False)
 before=tensor_hash(model.state_dict());mu=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None];sd=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
 def batch(subset,res):
  ims=[]
  for r in subset:
   with np.load(r['path'])as z:
    assert set(z.files)=={'rgb_static','rgb_gripper'}
    rgb=z[r['key']];assert rgb.dtype==np.uint8
   assert hashlib.sha256(rgb.tobytes()).hexdigest()==r['RGB_sha256']
   ims.append(F.interpolate(torch.from_numpy(rgb).permute(2,0,1)[None].float(),size=(res,res),mode='bilinear',align_corners=False)[0]/255)
  return(torch.stack(ims).cuda()-mu)/sd
 results=[]
 for res in(224,448):
  grid=res//14;data=batch(rows[:8],res);times=[]
  with torch.inference_mode():
   warm=model(pixel_values=data).last_hidden_state[:,1:];assert warm.shape==(8,grid*grid,384);del warm
   torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
   for _ in range(3):
    start=time.perf_counter();out=model(pixel_values=data).last_hidden_state[:,1:];torch.cuda.synchronize();times.append(time.perf_counter()-start);del out
  benchpeak=torch.cuda.max_memory_allocated();del data;gc.collect();torch.cuda.empty_cache()
  cache=[];torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
  for i in range(0,len(rows),8):
   data=batch(rows[i:i+8],res)
   with torch.inference_mode():tok=model(pixel_values=data).last_hidden_state[:,1:]
   cache.append(tok.cpu().numpy());del data,tok
  torch.cuda.synchronize();elapsed=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated()
  dest=E/f'data/features/r{res}.npz';dest.parent.mkdir(exist_ok=True);t=np.concatenate(cache)
  assert t.shape==(40,grid*grid,384)and np.isfinite(t).all()
  np.savez_compressed(dest,tokens=t,image_ids=np.array([r['image_id']for r in rows]))
  results.append({'resolution':res,'grid':grid,'images':40,'shape':list(t.shape),'raw_token_bytes':t.nbytes,'compressed_cache_bytes':dest.stat().st_size,'encoder_batch8_seconds':times,'encoder_batch8_median_seconds':float(np.median(times)),'encoder_benchmark_peak_allocated_bytes':benchpeak,'cache_loop_seconds_excluding_model_load_and_compression':elapsed,'cache_loop_peak_allocated_bytes':peak})
  print('RESOLUTION_FEATURE_CACHE_COMPLETE',res,round(elapsed,3),peak,flush=True);del cache,t;gc.collect();torch.cuda.empty_cache()
 assert tensor_hash(model.state_dict())==before
 save(E/'results/features.json',{'utc':now(),'encoder_hash':before,'encoder_unchanged':True,'encoder_updates':0,'annotation_reads_during_encoding':0,'RGB_only':True,'device':torch.cuda.get_device_name(0),'runs':results})
if __name__=='__main__':main()
