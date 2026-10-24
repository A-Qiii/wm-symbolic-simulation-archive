from config import *
from model import *
import time,fcntl,subprocess,sys,gc
def main():
 verify();assert not(E/'results/training.json').exists();cfg=read(E/'configs/protocol.json');lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 def guard(event,args):
  if event=='open'and isinstance(args[0],(str,bytes)):
   p=Path(args[0]).resolve()
   if p.is_relative_to(EX)and not p.is_relative_to(E):raise RuntimeError('Trainer cannot read prior or evaluation data: '+str(p))
   if p.is_relative_to(E/'data/evaluation'):raise RuntimeError('Trainer cannot read validation or D')
 sys.addaudithook(guard);torch.set_num_threads(4);torch.use_deterministic_algorithms(True);startup=time.perf_counter();torch.cuda.init();torch.cuda.synchronize();cuda_init=time.perf_counter()-startup
 monitorfile=(E/'logs/gpu_utilization.csv').open('w');monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total','--format=csv,noheader,nounits','-lms','250'],stdout=monitorfile,stderr=subprocess.DEVNULL);runs=[];hashes={}
 try:
  for seed in [0,1]:
   for form in FORMS:
    t0=time.perf_counter();data=np.load(E/'data/train/states.npz');norm=np.load(E/'configs'/f'normalization_{form}.npz');x=((features(data['z'],form)-norm['mean'])/norm['std']).astype(np.float32);sc=np.load(E/'configs'/f'schedule_seed{seed}.npz')['indices'];X=torch.tensor(x,device='cuda');Y=torch.tensor(data['y'],device='cuda');V=torch.tensor(data['v'],device='cuda');S=torch.tensor(sc,device='cuda');weight=torch.tensor(cfg['positive_weight'],device='cuda');torch.cuda.synchronize();prep=time.perf_counter()-t0
    torch.manual_seed(17078000+seed);q=make(form).cuda();opt=torch.optim.AdamW(q.parameters(),lr=.001,weight_decay=.001);base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();start_event=torch.cuda.Event(enable_timing=True);end_event=torch.cuda.Event(enable_timing=True);torch.cuda.synchronize();started_utc=now();start=time.perf_counter();start_event.record();history=[]
    for u in range(600):
     ix=S[u];ls=loss(q,X[ix],Y[ix],V[ix],weight);opt.zero_grad(set_to_none=True);ls.backward();gn=torch.nn.utils.clip_grad_norm_(q.parameters(),1);opt.step()
     if u+1 in [1,150,300,450,600]:history.append({'update':u+1,'loss':float(ls.detach()),'preclip_norm':float(gn)})
    end_event.record();torch.cuda.synchronize();elapsed=time.perf_counter()-start;end_utc=now();peak=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved();state={k:v.detach().cpu()for k,v in q.state_dict().items()};path=E/'models'/f'{form}_seed{seed}_update0600.pt';torch.save({'state':state,'form':form,'seed':seed,'updates':600},path);hashes[str(path)]=sha(path)
    run={'seed':seed,'form':form,'trainable_parameters':sum(p.numel()for p in q.parameters()),'updates':600,'batch_states':32,'object_examples_per_batch':96,'device':'cuda','cuda_device':torch.cuda.get_device_name(0),'data_prepare_transfer_seconds':prep,'training_wall_seconds':elapsed,'cuda_event_span_ms':start_event.elapsed_time(end_event),'cuda_event_note':'Elapsed span includes gaps from CPU dispatch; not the sum of active kernel times','training_started_utc':started_utc,'training_finished_utc':end_utc,'base_allocated_bytes':base,'peak_allocated_bytes':peak,'peak_reserved_bytes':reserved,'peak_above_base_bytes':peak-base,'history':history};runs.append(run);print('HOLDING_TRAINED',json.dumps(run),flush=True)
    del q,opt,X,Y,V,S,weight;gc.collect()
 finally:
  monitor.terminate();monitor.wait(timeout=10);monitorfile.close()
 save(E/'results/training.json',{'utc':now(),'runs':runs,'cuda_initialization_seconds':cuda_init,'gpu_sampling_interval_ms':250,'monitor_scope':'device-wide instantaneous utilization, same serial GPU phase; not per-kernel occupancy','model_fits':4,'extra_updates':0,'world_model_query_encoder_updates':0,'validation_access':False});save(E/'configs/final_heads_frozen.json',{'utc':now(),'before_evaluation':True,'sha256':hashes})
if __name__=='__main__':main()
