from common import *
from net import *
import time,fcntl,sys
def main():
 frozen=read(E/'results/checkpoints_frozen.json');assert not(E/'results/predictions_frozen.json').exists();opened=install_guard([BASE/f'data/features/{s}.npz'for s in ['train','dev']]);normal=np.load(E/'configs/normalization.npz');mu,sd=normal['mean'],normal['std'];torch.set_num_threads(4)
 def local_guard(event,args):
  if event=='open'and isinstance(args[0],(str,bytes)):
   p=Path(os.fsdecode(args[0])).resolve()
   if p.is_relative_to(E/'data')and not p.is_relative_to(E/'data/predictions'):raise RuntimeError('Inference denied labels/baseline evaluation files')
 sys.addaudithook(local_guard);lock=(P/'runs/gpu.lock').open('r+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.perf_counter();models=[]
 for seed in [0,1]:
  p=E/f'models/final_seed{seed}.pt';assert sha(p)==frozen['fits'][seed]['sha256'];m=make().cuda().eval();m.load_state_dict(torch.load(p,map_location='cuda',weights_only=True)['state']);models.append(m)
 for split in ['train','dev']:
  path=BASE/f'data/features/{split}.npz';assert sha(path)==read(E/'configs/input_frozen.json')['sha256'][str(path)];z=np.load(path);x=(z['x']-mu)/sd;ids=z['image_ids'];assert np.array_equal(ids,np.arange(256)if split=='train'else np.arange(256,384))
  for seed,m in enumerate(models):
   out=[]
   for j in range(0,len(x),16):
    with torch.inference_mode():raw=m(torch.tensor(x[j:j+16],device='cuda')).cpu().numpy()
    out.append(raw)
   save_raw_npz(E/f'data/predictions/raw_seed{seed}_{split}.npz',raw=np.concatenate(out),image_ids=ids);print('GAMMA0_PREDICTED',seed,split,flush=True)
 files=list((E/'data/predictions').glob('*.npz'));save(E/'results/predictions_frozen.json',{'utc':now(),'seconds':time.perf_counter()-start,'sha256':{str(p):sha(p)for p in files},'before_new_GT_scoring':True,'RGB_features_only_network_input':True,'old_files_opened':sorted(opened),'D_reads':0,'label_container_reads':0});lock.close()
if __name__=='__main__':main()
