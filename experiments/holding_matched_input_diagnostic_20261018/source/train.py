from common import *
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch,fcntl,time
def main():
 assert not(E/'results/training_attempts.json').exists();freeze=read(E/'configs/pretraining_frozen.json')
 for p,h in freeze['sha256'].items():assert sha(p)==h
 guard('train');torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 lock=(P/'runs/gpu.lock').open('r+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 arr=np.load(B/'data/features/train.npz');targets=read(B/'data/supervision/train_targets.json');assert np.array_equal(arr['record_indices'],[t['record_index']for t in targets]);y=np.array([t['y']for t in targets],np.float32);assert len(y)==920 and y.sum()==160
 norm=np.load(E/'configs/train_normalization.npz');x=((arr['x_pair']-norm['pair_mean'])/norm['pair_std']).astype(np.float32);yy=torch.from_numpy(y).cuda();fits=[];attempts=[]
 features={arm:torch.from_numpy(masked(x,arm)).cuda()for arm in ARMS}
 for seed in [0,1]:
  schedule=np.load(E/f'configs/schedule_seed{seed}.npz');assert schedule['indices'].shape==(600,32);assert np.array_equal(schedule['record_indices'],arr['record_indices'][schedule['indices']]);initial=torch.load(E/f'models/initial_pair_seed{seed}.pt',map_location='cpu',weights_only=True)['state']
  for arm in ARMS:
   attempts.append({'seed':seed,'arm':arm,'status':'started','utc':now()});save(E/'results/training_attempts.json',attempts);torch.manual_seed(261019210+seed);torch.cuda.manual_seed_all(261019210+seed);model=make().cuda();model.load_state_dict(initial);before=tensor_hash(model.state_dict());assert before==tensor_hash(initial);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001);criterion=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(4.75,device='cuda'));curve=[];start=time.perf_counter()
   for step,ids in enumerate(schedule['indices'],1):
    index=torch.from_numpy(ids).cuda();opt.zero_grad(set_to_none=True);loss=criterion(model(features[arm][index]).squeeze(1),yy[index]);assert torch.isfinite(loss);loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
    if step%50==0:curve.append({'step':step,'weighted_BCE':float(loss.detach()),'preclip_grad_norm':float(gn)})
   f=E/f'models/final_{arm}_seed{seed}.pt';torch.save({'state':model.state_dict(),'arm':arm,'seed':seed,'updates':600},f);finalhash=tensor_hash(model.state_dict());equiv=None
   if arm=='full':equiv=finalhash==tensor_hash(torch.load(B/f'models/final_pair_seed{seed}.pt',map_location='cpu',weights_only=True)['state'])
   fits.append({'seed':seed,'arm':arm,'updates':600,'parameters':8680,'initial_tensor_hash':before,'final_tensor_hash':finalhash,'file_sha256':sha(f),'schedule_sha256':sha(E/f'configs/schedule_seed{seed}.npz'),'seconds':time.perf_counter()-start,'full_original_pair_tensor_bitwise_equal':equiv,'curve':curve});attempts[-1]['status']='completed';save(E/'results/training_attempts.json',attempts);save(E/'results/training.json',{'utc':now(),'fits':fits,'dev_or_private_reads':0});print('FIT',arm,seed,'full_equivalent',equiv,flush=True)
 save(E/'configs/models_frozen.json',{'utc':now(),'sha256':{str(p):sha(p)for p in(E/'models').glob('final_*.pt')},'before_dev_prediction':True,'fits':6});lock.close()
if __name__=='__main__':main()
