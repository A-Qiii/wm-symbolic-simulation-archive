from common import *
from net import *
import sys,time,fcntl

def main():
 assert not(E/'results/predictions_frozen.json').exists();protocol=read(E/'configs/protocol.json');assert read(E/'results/features.json')['encoder_unchanged'];assert all(sha(p)==h for p,h in protocol['models']['sha256'].items());assert sha(E/'configs/normalization.npz')==protocol['models']['normalization_sha256'];install_guard([])
 def guard(event,args):
  if event=='open'and isinstance(args[0],(str,bytes)):
   p=Path(os.fsdecode(args[0])).resolve()
   if p.is_relative_to(E/'data/annotation_private')or p.is_relative_to(E/'data/calibration'):raise RuntimeError('Inference may not read new GT or calibration candidates')
 sys.addaudithook(guard);lock=(P/'runs/gpu.lock').open('r+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);d=E/'data/predictions';d.mkdir();normal=np.load(E/'configs/normalization.npz');z=np.load(E/'data/features/fresh.npz');x=(z['x']-normal['mean'])/normal['std'];ids=z['image_ids'];assert np.array_equal(ids,np.arange(256));start=time.perf_counter()
 for seed in [0,1]:
  m=make().cuda().eval();state=torch.load(E/f'models/final_seed{seed}.pt',map_location='cuda',weights_only=True);m.load_state_dict(state['state']);out=[]
  for i in range(0,len(x),16):
   with torch.inference_mode():out.append(m(torch.tensor(x[i:i+16],device='cuda')).cpu().numpy())
  save_raw_npz(d/f'raw_seed{seed}_fresh.npz',raw=np.concatenate(out),image_ids=ids);print('FRESH_PREDICTED',seed,flush=True)
 save(E/'results/predictions_frozen.json',{'utc':now(),'seconds':time.perf_counter()-start,'passes_per_seed':1,'new_GT_or_calibration_data_reads':0,'before_new_GT_scoring':True,'new_training_updates':0,'sha256':{str(p):sha(p)for p in d.glob('*.npz')}});lock.close()
if __name__=='__main__':main()
