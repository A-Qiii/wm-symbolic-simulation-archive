from common import *
from visual import Encoder,build_vectors
import torch,fcntl,time,shutil,sys,subprocess

def main():
 assert not(E/'data/public/vectors.npz').exists();rows=read(E/'data/public/states.json')
 for p,h in read(E/'configs/pre_encoding_frozen.json')['sha256'].items():assert sha(p)==h,p
 for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,E/'source'/p.name)
 save(E/'configs/encoding_frozen.json',{'utc':now(),'before_feature_extraction':True,'code_sha256':{str(p):sha(p)for p in(E/'source').glob('*.py')},'DINO_sha256':sha(DINO/'model.safetensors'),'schema':['rgb_static','rgb_gripper'],'PCA_fit':'training union RGB only','ROIs':ROIS})
 def guard(event,args):
  if event=='open'and isinstance(args[0],(str,bytes)):
   p=Path(args[0]).resolve()
   if p.is_relative_to(EX)and not p.is_relative_to(E):raise RuntimeError('Visual extraction cannot read previous experiment payloads')
   if p.is_relative_to(E/'data')and not p.is_relative_to(E/'data/public'):raise RuntimeError('Visual extraction cannot read training labels or private data')
 sys.addaudithook(guard)
 lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4)
 torch.cuda.init();torch.cuda.reset_peak_memory_stats();start=time.perf_counter();enc=Encoder();before=tensor_hash(enc.net.state_dict());cache={};out=[]
 monitorfile=(E/'logs/encoding_gpu.csv').open('w');monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used','--format=csv,noheader,nounits','-lms','250'],stdout=monitorfile,stderr=subprocess.DEVNULL)
 try:
  unique=[];seen=set()
  for r in rows:
   if r['RGB_content_sha256']not in seen:unique.append(r);seen.add(r['RGB_content_sha256'])
  for i in range(0,len(unique),8):
   rr=unique[i:i+8];ff=enc.encode([r['path']for r in rr])
   for r,f in zip(rr,ff):cache[r['RGB_content_sha256']]=f
   if(i+8)%240==0:print('PURE_RGB_ENCODED',i+8,'/',len(unique),flush=True)
  assert before==tensor_hash(enc.net.state_dict());torch.cuda.synchronize();encseconds=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated();del enc;torch.cuda.empty_cache()
  for r in rows:out.append(cache[r['RGB_content_sha256']])
  raw={k:np.stack([x[k]for x in out])for k in out[0]};np.savez_compressed(E/'data/public/raw_features.npz',**raw)
  train=np.array([r['train']for r in rows]);vis=raw['geometry'][train,...,4].astype(bool)
  obj=raw['objects'][train][vis].reshape(-1,384);end=raw['endpoints'][train].reshape(-1,384);tokens=np.concatenate([obj,end]).astype(np.float32)
  mean=tokens.mean(0);std=np.maximum(tokens.std(0),.05);x=(tokens-mean)/std;cov=x.T@x/max(1,len(x)-1);val,vec=np.linalg.eigh(cov);order=np.argsort(val)[::-1];basis=vec[:,order[:32]]
  for j in range(32):
   if basis[np.argmax(abs(basis[:,j])),j]<0:basis[:,j]*=-1
  scale=np.maximum((x@basis).std(0),.1);norm={'mean':mean,'std':std,'basis':basis,'scale':scale};np.savez_compressed(E/'data/public/vision_projection.npz',**norm)
  vectors=build_vectors(raw,norm);np.savez_compressed(E/'data/public/vectors.npz',**vectors)
  save(E/'results/encoding.json',{'utc':now(),'RGB_states':len(rows),'unique_RGB_encoded':len(unique),'DINO_frozen_tensor_hash':before,'feature_seconds':encseconds,'total_with_PCA_seconds':time.perf_counter()-start,'peak_allocated_bytes':peak,'PCA_training_states':int(train.sum()),'PCA_training_tokens':len(tokens),'PCA32_variance_fraction':float(val[order[:32]].sum()/val.sum()),'private_payload_reads':0,'robot_channels':0,'old_latent_or_old_PCA_loaded':False,'vision_only':True,'whole_image_DINO_context':True,'endpoint_detection':'fixed pixel-camera regions, no simulator lookup','object_visible_per_color_camera':raw['geometry'][...,4].sum(0).tolist()})
 finally:monitor.terminate();monitor.wait(timeout=10);monitorfile.close()
 print('PURE_VISUAL_FEATURES_COMPLETE',len(rows),flush=True)
if __name__=='__main__':main()
