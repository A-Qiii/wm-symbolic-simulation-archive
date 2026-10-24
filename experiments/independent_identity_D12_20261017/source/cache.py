from config import *
import sys,torch,fcntl,argparse,time
sys.path.insert(0,str(REP/'source'))
from features import Encoder
from representation import transform
from runtime import load
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['combo'],required=True);a=ap.parse_args();verify_frozen();d=E/'data'/a.phase;rows=read(d/'public/states.json');lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4)
 def guard(event,args):
  if event=='open'and isinstance(args[0],(str,bytes))and('/_private/'in str(args[0])):raise RuntimeError('Feature encoder cannot open labels/private trace')
 sys.addaudithook(guard);start=time.monotonic();enc=Encoder();fs=[]
 for k in range(0,len(rows),8):
  rr=rows[k:k+8];paths=[r['path']for r in rr];f=enc.encode(paths+[paths[-1]]*(8-len(paths)))[:len(paths)]
  for row,ff in zip(rr,f):np.savez_compressed(d/'public/features'/f'{row["id"]}.npz',**ff);fs.append(ff)
  if k%160==0:print('CACHE',a.phase,k+len(rr),'/',len(rows),round(time.monotonic()-start,1),'sec',flush=True)
 del enc;torch.cuda.empty_cache();arrays={key:np.stack([f[key]for f in fs]).astype(np.float32)for key in fs[0]};norm=dict(np.load(REP/'data/public/normalization.npz'));o,s=transform(arrays,norm);m,q,stats=load(0,'canonical_matched');zs=[]
 with torch.inference_mode():
  for k in range(0,len(o),128):zs.append(m.encode(torch.tensor(o[k:k+128],device='cuda'),torch.tensor(s[k:k+128],device='cuda')).cpu().numpy())
 z=np.concatenate(zs);np.savez_compressed(d/'public/inputs.npz',objects=o,scene=s,z=z);save(E/'results'/f'cache_{a.phase}.json',{'states':len(rows),'seconds':time.monotonic()-start,'frozen_DINO':True,'frozen_normalization':True,'frozen_encoder_sha256':sha(REP/'models/canonical_matched_seed0/update1000.pt'),'private_reads':0,'fixed_feature_batch_size':8,'last_batch_padded':True});print('CACHE_COMPLETE',a.phase,len(rows),flush=True)
if __name__=='__main__':main()
