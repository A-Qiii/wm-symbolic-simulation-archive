from common import *
from net import *
import sys,time,fcntl,gc

def main():
 assert not(E/'results/training_attempts.json').exists()
 for p,h in read(E/'configs/training_frozen.json')['sha256'].items():assert sha(p)==h,p
 def guard(event,args):
  if event=='open'and isinstance(args[0],(str,bytes)):
   p=Path(args[0]).resolve()
   if p.is_relative_to(P/'experiments')and not p.is_relative_to(E):raise RuntimeError('No other experiments during training')
   if p.is_relative_to(E/'data')and not p.is_relative_to(E/'data/train'):raise RuntimeError('No evaluation, raw RGB, or annotation access in fitting')
 sys.addaudithook(guard);lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.cuda.init();d=np.load(E/'data/train/train.npz');X,Y,V,B=[torch.tensor(d[k],device='cuda')for k in['x','y','v','b']];attempts=[];runs=[];heads={}
 for seed in(0,1):
  q=make().cuda();ini=torch.load(E/f'models/initial_seed{seed}.pt',weights_only=True,map_location='cpu');q.load_state_dict(ini['state']);assert tensor_hash(q.state_dict())==ini['initial_hash'];S=torch.tensor(np.load(E/f'configs/schedule_seed{seed}.npy'),device='cuda');opt=torch.optim.AdamW(q.parameters(),lr=.001,weight_decay=.0001);attempts.append({'seed':seed,'status':'started','utc':now()});save(E/'results/training_attempts.json',attempts);history=[];torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
  for u in range(600):
   ix=S[u];ls,components=loss(q(X[ix]),Y[ix],V[ix],B[ix]);assert torch.isfinite(ls);opt.zero_grad(set_to_none=True);ls.backward();gn=torch.nn.utils.clip_grad_norm_(q.parameters(),1);opt.step()
   if u+1 in[1,150,300,450,600]:history.append({'update':u+1,'total':float(ls.detach()),'focal_smoothL1_GIoUloss':components.cpu().tolist(),'gradient_norm':float(gn)})
  torch.cuda.synchronize();sec=time.perf_counter()-start;state={k:v.detach().cpu()for k,v in q.state_dict().items()};dest=E/f'models/localizer_seed{seed}_update0600.pt';torch.save({'state':state,'initial_hash':ini['initial_hash'],'final_hash':tensor_hash(state),'updates':600},dest);heads[str(dest)]=sha(dest);attempts[-1]['status']='completed';runs.append({'seed':seed,'updates':600,'parameters':25290,'seconds':sec,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'history':history});save(E/'results/training_attempts.json',attempts);save(E/'results/training_progress.json',runs);print('LOCALIZER_FIT_COMPLETED',seed,round(sec,3),flush=True);del q,opt;gc.collect()
 save(E/'results/training.json',{'utc':now(),'fits':2,'runs':runs,'DINO_updates':0,'validation_access':False,'model_selection':False,'device':torch.cuda.get_device_name(0)});save(E/'configs/final_heads_frozen.json',{'utc':now(),'before_evaluation':True,'sha256':heads});print('TWO_LOCALIZER_FITS_COMPLETE',flush=True)
if __name__=='__main__':main()
