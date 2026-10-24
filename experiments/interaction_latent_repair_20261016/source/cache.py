from common import *
from features import Encoder
from representation import transform
import torch,fcntl,time,argparse
def encode_paths(paths,encoder):
 out=[]
 for i,path in enumerate(paths):
  out.append(encoder.encode([path]*8)[0])
  if i%20==0:print('ENCODED',i+1,'/',len(paths),flush=True)
 f={k:np.stack([r[k]for r in out])for k in out[0]};norm=dict(np.load(E/'data/public/normalization.npz'));return transform(f,norm)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--D2',action='store_true');a=ap.parse_args();lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);start=time.monotonic();enc=Encoder()
 if not a.D2:
  states=read(E/'data/public/states.json');n=read(E/'results/preparation.json')['old_states'];o,s=encode_paths([r['path']for r in states[n:]],enc)
  old=dict(np.load(E/'data/public/old_inputs.npz'));o=np.concatenate([old['objects'],o]);s=np.concatenate([old['scene'],s]);dest=E/'data/public/inputs.npz'
 else:
  states=read(E/'data/D2/public/states.json');o,s=encode_paths([r['path']for r in states],enc);dest=E/'data/D2/public/inputs.npz'
 np.savez_compressed(dest,objects=o,scene=s);save(E/'results'/('D2_cache.json'if a.D2 else'cache.json'),{'states':len(o),'seconds':time.monotonic()-start,'fixed_DINO_batch8_duplicate_current':True,'normalization_sha256':sha(E/'data/public/normalization.npz'),'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated()});print('CACHE_COMPLETE',dest,flush=True)
if __name__=='__main__':main()
