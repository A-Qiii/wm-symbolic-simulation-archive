from common import *
from net import *
import sys,time,fcntl
def main():
 frozen=read(E/'results/checkpoints_frozen.json');assert not(E/'results/predictions_frozen.json').exists();rows=read(E/'data/public/images.json');normal=np.load(E/'configs/normalization.npz');mu=normal['mean'];sd=normal['std'];dest=E/'data/predictions';dest.mkdir();torch.set_num_threads(4)
 def guard(event,args):
  if event=='open'and isinstance(args[0],(str,bytes)):
   p=Path(args[0]).resolve()
   if p.is_relative_to(E/'data/annotation_private')or p.is_relative_to(E/'data/train'):raise RuntimeError('Inference denied label-containing data')
   if p.is_relative_to(P/'experiments')and not p.is_relative_to(E):raise RuntimeError('Inference denied old experiments')
 sys.addaudithook(guard);lock=(P/'runs/gpu.lock').open('r+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.perf_counter();models=[];out={0:[],1:[]}
 for seed in [0,1]:
  path=E/f'models/final_seed{seed}.pt';assert sha(path)==frozen['fits'][seed]['sha256'];m=make().cuda().eval();m.load_state_dict(torch.load(path,map_location='cuda',weights_only=True)['state']);models.append(m)
 for split in ['train','dev','final']:
  z=np.load(E/f'data/features/{split}.npz');x=(z['x']-mu)/sd;ids=z['image_ids']
  for seed,m in enumerate(models):
   raw=[]
   for j in range(0,len(x),16):
    with torch.inference_mode():r=m(torch.tensor(x[j:j+16],device='cuda')).cpu().numpy()
    raw.append(r)
   raw=np.concatenate(raw);save_raw_npz(dest/f'raw_seed{seed}_{split}.npz',raw=raw,image_ids=ids)
   for i,rr in zip(ids,raw):out[seed].append({'image_id':int(i),'predictions':decode(rr)})
   print('PREDICTED',split,seed,flush=True)
 for seed in [0,1]:save(dest/f'decoded_seed{seed}.json',out[seed])
 save(E/'results/predictions_frozen.json',{'utc':now(),'seconds':time.perf_counter()-start,'final_inference_passes_per_seed':1,'GT_reads':0,'head_inputs':'RGB-derived normalized frozen DINO tokens only','sha256':{str(p):sha(p)for p in dest.glob('*')}});lock.close()
if __name__=='__main__':main()
