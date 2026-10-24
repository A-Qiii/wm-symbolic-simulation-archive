from common import *
import torch,fcntl,time,sys,subprocess,gc

def main():
 assert not(E/'results/training_attempts.json').exists()
 for p,h in read(E/'configs/training_frozen.json')['sha256'].items():assert sha(p)==h,p
 cfg=read(E/'configs/protocol.json');lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 def guard(event,args):
  if event=='open'and isinstance(args[0],(str,bytes)):
   p=Path(args[0]).resolve()
   if p.is_relative_to(EX)and not p.is_relative_to(E):raise RuntimeError('No other experiments during training')
   if p.is_relative_to(E/'data')and not p.is_relative_to(E/'data/train'):raise RuntimeError('Training input restricted to frozen training arrays')
 sys.addaudithook(guard);torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.cuda.init()
 monitorfile=(E/'logs/training_gpu.csv').open('w');monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used','--format=csv,noheader,nounits','-lms','250'],stdout=monitorfile,stderr=subprocess.DEVNULL)
 attempts=[];runs=[];hashes={}
 try:
  for head in HEADS:
   data=np.load(E/f'data/train/{head}.npz');assert np.count_nonzero(data['x'][:,:,74:])==0
   X=torch.tensor(data['x'],device='cuda');Y=torch.tensor(data['y'],device='cuda');V=torch.tensor(data['v'],device='cuda');weight=torch.tensor(cfg['positive_weights'][head],device='cuda')
   for seed in(0,1):
    sc=np.load(E/f'configs/schedule_{head}_seed{seed}.npz');schedule=np.concatenate([sc['common_old'],sc['official']+cfg['old_pool_states'][head]],1);S=torch.tensor(schedule,device='cuda')
    q=make().cuda();initial=torch.load(E/f'models/initial_{head}_seed{seed}.pt',weights_only=True,map_location='cpu');q.load_state_dict(initial['state']);assert tensor_hash(q.state_dict())==initial['initial_hash'];opt=torch.optim.AdamW(q.parameters(),lr=.001,weight_decay=.001)
    attempts.append({'head':head,'arm':'zero_context','seed':seed,'status':'started','utc':now()});save(E/'results/training_attempts.json',attempts)
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter();hist=[]
    for u in range(600):
     ix=S[u];logits=q(X[ix]).squeeze(-1);loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,Y[ix],pos_weight=weight,reduction='none')*V[ix];ls=(loss.sum(1)/V[ix].sum(1).clamp(min=1)).mean();opt.zero_grad(set_to_none=True);ls.backward();gn=torch.nn.utils.clip_grad_norm_(q.parameters(),1);opt.step()
     if u+1 in[1,150,300,450,600]:hist.append({'update':u+1,'loss':float(ls.detach()),'preclip_norm':float(gn)})
    torch.cuda.synchronize();seconds=time.perf_counter()-start;state={k:v.detach().cpu()for k,v in q.state_dict().items()};dest=E/f'models/{head}_zero_context_seed{seed}_update0600.pt';torch.save({'state':state,'initial_hash':initial['initial_hash'],'final_hash':tensor_hash(state),'updates':600},dest);hashes[str(dest)]=sha(dest)
    runs.append({'head':head,'arm':'zero_context','seed':seed,'updates':600,'parameters':6689,'seconds':seconds,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'initial_hash':initial['initial_hash'],'final_hash':tensor_hash(state),'history':hist});attempts[-1]['status']='completed';save(E/'results/training_attempts.json',attempts);save(E/'results/training_progress.json',runs);print('MATCHED_ZERO_FIT',head,seed,round(seconds,3),flush=True);del q,opt;gc.collect()
   del X,Y,V,weight,S;gc.collect()
 finally:monitor.terminate();monitor.wait(timeout=10);monitorfile.close()
 assert len(runs)==4;save(E/'results/training.json',{'utc':now(),'runs':runs,'fits':4,'encoder_T_updates':0,'validation_access':False,'device':torch.cuda.get_device_name(0)});save(E/'configs/final_heads_frozen.json',{'utc':now(),'before_evaluation':True,'sha256':hashes});print('FOUR_MATCHED_ZERO_FITS_COMPLETE',flush=True)
if __name__=='__main__':main()
