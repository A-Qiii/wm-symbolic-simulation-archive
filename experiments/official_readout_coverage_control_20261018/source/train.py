from coverage_common import *
import torch,fcntl,time,sys,subprocess,gc
def main():
 assert not(E/'results/training_attempts.json').exists();frozen=read(E/'configs/training_frozen.json')
 for p,h in frozen['sha256'].items():assert sha(p)==h,p
 cfg=read(E/'configs/protocol.json');lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 def guard(event,args):
  if event=='open'and isinstance(args[0],(str,bytes)):
   p=Path(args[0]).resolve()
   if p.is_relative_to(EX)and not p.is_relative_to(E):raise RuntimeError('Trainer cannot access previous or evaluation experiments')
   if p.is_relative_to(E/'data')and not p.is_relative_to(E/'data/train'):raise RuntimeError('Trainer cannot read public/evaluation/private pools')
 sys.addaudithook(guard);torch.set_num_threads(4);torch.use_deterministic_algorithms(True);t0=time.perf_counter();torch.cuda.init();torch.cuda.synchronize();cuda_init=time.perf_counter()-t0
 monitorfile=(E/'logs/training_gpu.csv').open('w');monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used','--format=csv,noheader,nounits','-lms','250'],stdout=monitorfile,stderr=subprocess.DEVNULL);runs=[];attempts=[];hashes={}
 try:
  official=np.load(E/'data/train/official_train.npz',allow_pickle=False)
  for head in HEADS:
   old=np.load(E/'data/train'/f'old_{head}.npz',allow_pickle=False);norm=np.load(E/'configs'/f'normalization_{head}.npz',allow_pickle=False);x=np.concatenate([old['x'],((official['z'][:,:3]-norm['mean'])/norm['std']).astype(np.float32)]);g=slice(0,3)if head=='Holding'else slice(3,6);y=np.concatenate([old['y'][:,g],official['y'][:,g]]);v=np.concatenate([old['v'][:,g],official['v'][:,g]]);X=torch.tensor(x,device='cuda');Y=torch.tensor(y,device='cuda');V=torch.tensor(v,device='cuda');weight=torch.tensor(cfg['positive_weights'][head],device='cuda')
   for seed in[0,1]:
    sched=np.load(E/'configs'/f'schedule_{head}_seed{seed}.npz',allow_pickle=False);initial=torch.load(E/'models'/f'initial_{head}_seed{seed}.pt',map_location='cpu',weights_only=True)
    for arm in ARMS:
     sc=sched['control']if arm=='control'else np.concatenate([sched['common_old'],sched['official']+len(old['x'])],axis=1);S=torch.tensor(sc,device='cuda');q=make().cuda();q.load_state_dict(initial['state']);assert tensor_hash(q.state_dict())==initial['state_hash'];opt=torch.optim.AdamW(q.parameters(),lr=.001,weight_decay=.001)
     attempts.append({'head':head,'seed':seed,'arm':arm,'status':'started','utc':now()});save(E/'results/training_attempts.json',attempts);base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter();history=[];first=torch.cuda.Event(enable_timing=True);last=torch.cuda.Event(enable_timing=True);first.record()
     for u in range(600):
      ix=S[u];logits=q(X[ix]).squeeze(-1);loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,Y[ix],pos_weight=weight,reduction='none')*V[ix];ls=(loss.sum(1)/V[ix].sum(1).clamp(min=1)).mean();opt.zero_grad(set_to_none=True);ls.backward();gn=torch.nn.utils.clip_grad_norm_(q.parameters(),1);opt.step()
      if u+1 in [1,150,300,450,600]:history.append({'update':u+1,'loss':float(ls.detach()),'preclip_norm':float(gn)})
     last.record();torch.cuda.synchronize();elapsed=time.perf_counter()-start;state={k:v.detach().cpu()for k,v in q.state_dict().items()};path=E/'models'/f'{head}_{arm}_seed{seed}_update0600.pt';torch.save({'state':state,'head':head,'arm':arm,'seed':seed,'updates':600,'initial_hash':initial['state_hash'],'final_hash':tensor_hash(state)},path);hashes[str(path)]=sha(path)
     runs.append({'head':head,'arm':arm,'seed':seed,'updates':600,'parameters':2113,'initial_hash':initial['state_hash'],'final_hash':tensor_hash(state),'seconds':elapsed,'cuda_event_span_ms':first.elapsed_time(last),'base_allocated_bytes':base,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'history':history});attempts[-1]['status']='completed';save(E/'results/training_attempts.json',attempts);save(E/'results/training_progress.json',runs);print('READOUT_FIT',head,seed,arm,'seconds',round(elapsed,3),'loss',history[-1]['loss'],flush=True);del q,opt,S;gc.collect()
   del X,Y,V,weight;gc.collect()
 finally:monitor.terminate();monitor.wait(timeout=10);monitorfile.close()
 assert len(runs)==8;save(E/'results/training.json',{'utc':now(),'runs':runs,'fits':8,'updates_total':4800,'cuda_initialization_seconds':cuda_init,'device':torch.cuda.get_device_name(0),'encoder_T_drawer_updates':0,'validation_data_access':False,'GPU_monitor_interval_ms':250});save(E/'configs/final_heads_frozen.json',{'utc':now(),'before_actual_evaluation':True,'sha256':hashes});print('EIGHT_FITS_COMPLETE',flush=True)
if __name__=='__main__':main()
