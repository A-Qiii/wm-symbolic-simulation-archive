from common import *
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch,time,fcntl
def main():
 assert not(E/'results/training_attempts.json').exists()
 for p,h in read(E/'configs/pretraining_frozen.json')['sha256'].items():assert sha(p)==h
 guard('train')
 # Additional guard for new-context dev feature cache.
 def dev_guard(event,args):
  if event=='open'and isinstance(args[0],(str,bytes))and Path(os.fsdecode(args[0])).resolve()==E/'data/dev_features.npz':raise RuntimeError('No dev features at training')
 import sys;sys.addaudithook(dev_guard);torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;lock=(P/'runs/gpu.lock').open('r+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);arr=np.load(E/'data/train_features.npz');targets=read(B/'data/supervision/train_targets.json');assert np.array_equal(arr['record_indices'],[t['record_index']for t in targets]);y=np.array([t['y']for t in targets],np.float32);assert len(y)==920 and y.sum()==160;mu=arr['x'].mean(0,dtype=np.float64).astype(np.float32);sd=np.maximum(arr['x'].std(0,dtype=np.float64),.05).astype(np.float32);old=np.load(B/'configs/train_normalization.npz');assert np.array_equal(mu[:261],old['pair_mean'])and np.array_equal(sd[:261],old['pair_std']);npz(E/'configs/train_normalization.npz',mean=mu,std=sd);x=((arr['x']-mu)/sd).astype(np.float32);features={a:torch.from_numpy(masked(x,a)).cuda()for a in ARMS};yy=torch.from_numpy(y).cuda();fits=[];attempts=[]
 for seed in [0,1]:
  schedule=np.load(E/f'configs/schedule_seed{seed}.npz');assert np.array_equal(schedule['record_indices'],arr['record_indices'][schedule['indices']]);initial=torch.load(E/f'models/initial_seed{seed}.pt',map_location='cpu',weights_only=True)['state']
  for arm in ARMS:
   attempts.append({'seed':seed,'arm':arm,'status':'started','utc':now()});save(E/'results/training_attempts.json',attempts);torch.manual_seed(261019210+seed);torch.cuda.manual_seed_all(261019210+seed);model=make().cuda();model.load_state_dict(initial);before=tensor_hash(model.state_dict());opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001);criterion=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(4.75,device='cuda'));curve=[];start=time.perf_counter()
   for step,ids in enumerate(schedule['indices'],1):
    ii=torch.from_numpy(ids).cuda();opt.zero_grad(set_to_none=True);loss=criterion(model(features[arm][ii]).squeeze(1),yy[ii]);assert torch.isfinite(loss);loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
    if step%50==0:curve.append({'step':step,'weighted_BCE':float(loss.detach()),'preclip_grad_norm':float(gn)})
   f=E/f'models/final_{arm}_seed{seed}.pt';torch.save({'state':model.state_dict(),'arm':arm,'seed':seed,'updates':600},f);fits.append({'arm':arm,'seed':seed,'updates':600,'parameters':12904,'initial_tensor_hash':before,'final_tensor_hash':tensor_hash(model.state_dict()),'sha256':sha(f),'schedule_sha256':sha(E/f'configs/schedule_seed{seed}.npz'),'seconds':time.perf_counter()-start,'curve':curve});attempts[-1]['status']='completed';save(E/'results/training_attempts.json',attempts);save(E/'results/training.json',{'utc':now(),'fits':fits,'dev_and_private_reads':0});print('FIT',arm,seed,flush=True)
 save(E/'configs/models_frozen.json',{'utc':now(),'before_dev_prediction':True,'fits':4,'sha256':{str(p):sha(p)for p in(E/'models').glob('final_*.pt')},'normalization_sha256':sha(E/'configs/train_normalization.npz')});lock.close()
if __name__=='__main__':main()
