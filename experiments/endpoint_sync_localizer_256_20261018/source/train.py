import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from common import *
from net import *
import time,fcntl,sys
def main():
 assert read(E/'results/data_gate.json')['passed'];assert not(E/'results/checkpoints_frozen.json').exists();frozen=read(E/'configs/training_frozen.json');assert all(sha(p)==h for p,h in frozen['sha256'].items());torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 def guard(event,args):
  if event=='open'and isinstance(args[0],(str,bytes)):
   p=Path(args[0]).resolve()
   if p.is_relative_to(E/'data')and not p.is_relative_to(E/'data/train'):raise RuntimeError('Training denied non-train data')
   if p.is_relative_to(P/'experiments')and not p.is_relative_to(E):raise RuntimeError('Training denied old experiments')
 sys.addaudithook(guard);lock=(P/'runs/gpu.lock').open('r+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.cuda.reset_peak_memory_stats();z=np.load(E/'data/train/train.npz');data={k:torch.tensor(z[k],device='cuda')for k in ['x','y','b','v']};out=[]
 for seed in [0,1]:
  torch.manual_seed(seed);m=make().cuda();initial=torch.load(E/f'models/initial_seed{seed}.pt',map_location='cuda',weights_only=True);m.load_state_dict(initial['state']);assert tensor_hash(m.state_dict())==initial['initial_hash'];optimizer=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);schedule=np.load(E/f'configs/schedule_seed{seed}.npy');history=[];start=time.perf_counter()
  for step,ids in enumerate(schedule):
   ii=torch.tensor(ids,device='cuda');optimizer.zero_grad(set_to_none=True);ls,parts=loss(m(data['x'][ii]),data['y'][ii],data['v'][ii],data['b'][ii]);assert torch.isfinite(ls);ls.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);optimizer.step()
   if step==0 or (step+1)%100==0:history.append({'update':step+1,'total':float(ls),'focal':float(parts[0]),'smoothL1':float(parts[1]),'GIoU':float(parts[2])});print('TRAIN',seed,history[-1]if step==0 or(step+1)%100==0 else '',flush=True)if step==0 or(step+1)%100==0 else None
  torch.cuda.synchronize();dest=E/f'models/final_seed{seed}.pt';torch.save({'state':{k:v.cpu()for k,v in m.state_dict().items()},'seed':seed,'updates':1200},dest);result={'seed':seed,'updates':1200,'seconds':time.perf_counter()-start,'sha256':sha(dest),'parameter_hash':tensor_hash(m.state_dict()),'history':history};save(E/f'results/train_seed{seed}.json',result);out.append(result)
 save(E/'results/checkpoints_frozen.json',{'utc':now(),'both_fits_complete_before_any_dev_final_metrics':True,'private_or_eval_data_reads':0,'peak_GPU_bytes':torch.cuda.max_memory_allocated(),'fits':out});lock.close()
if __name__=='__main__':main()
