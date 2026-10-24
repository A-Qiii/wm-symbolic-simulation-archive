import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from common import *
from net import *
import time,fcntl,sys
def main():
 assert read(E/'results/preflight.json')['passed'];assert not(E/'results/checkpoints_frozen.json').exists();assert all(sha(p)==h for p,h in read(E/'configs/pre_run_frozen.json')['sha256'].items());opened=install_guard([BASE/'data/train/train.npz'])
 def local_guard(event,args):
  if event=='open'and isinstance(args[0],(str,bytes))and Path(os.fsdecode(args[0])).resolve().is_relative_to(E/'data'):raise RuntimeError('Training may not open evaluation/baseline/label files')
 sys.addaudithook(local_guard);assert sha(BASE/'data/train/train.npz')==read(E/'configs/input_frozen.json')['sha256'][str(BASE/'data/train/train.npz')];torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;lock=(P/'runs/gpu.lock').open('r+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.cuda.reset_peak_memory_stats();z=np.load(BASE/'data/train/train.npz');data={k:torch.tensor(z[k],device='cuda')for k in ['x','y','b','v']};counts=z['y'].sum(1);fits=[]
 for seed in [0,1]:
  torch.manual_seed(seed);m=make().cuda();initial=torch.load(E/f'models/initial_seed{seed}.pt',map_location='cuda',weights_only=True);m.load_state_dict(initial['state']);assert tensor_hash(m.state_dict())==initial['initial_hash'];opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);schedule=np.load(E/f'configs/schedule_seed{seed}.npy');records=[];start=time.perf_counter();exposure=np.zeros(3,np.int64)
  for step,ids in enumerate(schedule):
   ii=torch.tensor(ids,device='cuda');opt.zero_grad(set_to_none=True);ls,parts=loss(m(data['x'][ii]),data['y'][ii],data['v'][ii],data['b'][ii]);assert torch.isfinite(ls);ls.backward();rows=torch.sqrt(m[-1].weight.grad.square().sum(1)+m[-1].bias.grad.square());gn=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);assert torch.isfinite(gn);opt.step();pos=counts[ids].sum(0).astype(int);exposure+=pos
   rec={'update':step+1,'total':float(ls.detach()),'classification':float(parts[0]),'smoothL1':float(parts[1]),'GIoU':float(parts[2]),'preclip_global_gradient_norm':float(gn),'clip_multiplier':min(1.,1./(float(gn)+1e-6)),'preclip_output_logit_gradient_norm_by_slot':rows[[0,5,10]].detach().cpu().tolist(),'positive_exposures_by_slot':pos.tolist()};records.append(rec)
   if step==0 or(step+1)%100==0:print('MIXED_TRAIN',seed,step+1,'loss',rec['total'],'cls',rec['classification'],'grad',rec['preclip_global_gradient_norm'],flush=True)
  torch.cuda.synchronize();dest=E/f'models/final_seed{seed}.pt';torch.save({'state':{k:v.cpu()for k,v in m.state_dict().items()},'seed':seed,'updates':1200,'gamma_by_slot':[0,2,2]},dest);assert exposure.tolist()==[14250,37425,825];fit={'seed':seed,'updates':1200,'seconds':time.perf_counter()-start,'sha256':sha(dest),'parameter_hash':tensor_hash(m.state_dict()),'history':records,'positive_exposures_by_slot':exposure.tolist(),'batches_containing_collision_positive':sum(r['positive_exposures_by_slot'][2]>0 for r in records),'clipped_updates':sum(r['clip_multiplier']<1 for r in records),'nonfinite_gradients':0};save(E/f'results/train_seed{seed}.json',fit);fits.append({k:v for k,v in fit.items()if k!='history'})
 save(E/'results/checkpoints_frozen.json',{'utc':now(),'two_fixed_fits_completed_before_eval':True,'fits':fits,'peak_GPU_bytes':torch.cuda.max_memory_allocated(),'old_files_opened':sorted(opened),'training_D_or_development_reads':0,'training_local_evaluation_reads':0,'new_runs':2});lock.close()
if __name__=='__main__':main()
