from common import *
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch,time,fcntl,signal,resource
from wm_symbolic.world_model import WorldModel
from fixed_dataset import CoreEpisodes
from wm_symbolic.checkpointing import learning_rate,atomic_save
from losses import losses,backward_routed,clip_groups
STOP=False
def signal_stop(sig,frame):
 global STOP
 STOP=True
def main():
 verify();assert read(E/'results/preflight.json')['passed'];guard('train')
 lock=(P/'runs/gpu.lock').open('r+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for s in [signal.SIGINT,signal.SIGTERM,signal.SIGUSR1]:signal.signal(s,signal_stop)
 torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('high')
 norm=read(E/'configs/normalization.json');data=CoreEpisodes(P,'train',normalizer=norm);cached=[data.get(i)for i in range(len(data.episodes))]
 # In-memory cache changes only I/O: tensor values exactly match the original loader.
 check=data.batch([0,1],'cpu');assert all(torch.equal(v,torch.stack([cached[0][k],cached[1][k]]))for k,v in check.items());del check
 w=[torch.tensor(v,device='cuda')for v in norm['class_weights']]
 attempts=read(E/'results/training_attempts.json')if(E/'results/training_attempts.json').exists()else[]
 for seed in [0,1]:
  schedule=np.load(E/f'configs/schedule_seed{seed}.npz');initial=torch.load(E/f'models/initial_seed{seed}.pt',map_location='cpu',weights_only=True)
  for arm in ARMS:
   folder=E/'models'/f'{arm}_seed{seed}';folder.mkdir(exist_ok=True);resultfile=E/'results'/f'train_{arm}_seed{seed}.json'
   if resultfile.exists():assert read(resultfile)['updates']==6000;continue
   m=WorldModel(skill_count=6,event_count=3).cuda();m.load_state_dict(initial['state']);assert th(m.state_dict())==initial['hash']
   opt=torch.optim.AdamW(m.parameters(),lr=3e-4,betas=(.9,.999),weight_decay=1e-4);start=0
   if(folder/'latest.pt').exists():
    ck=torch.load(folder/'latest.pt',map_location='cuda',weights_only=False);assert ck['arm']==arm and ck['seed']==seed and ck['protocol_sha256']==sha(E/'configs/protocol.json');m.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);start=ck['update'];del ck
   attempts.append({'arm':arm,'seed':seed,'start_update':start,'status':'running','utc':now(),'pid':os.getpid(),'size':SIZE});save(E/'results/training_attempts.json',attempts)
   torch.cuda.reset_peak_memory_stats();begin=time.perf_counter();log=(E/'logs'/f'train_{arm}_seed{seed}.jsonl').open('a');m.train()
   def checkpoint(u):return {'model':m.state_dict(),'optimizer':opt.state_dict(),'update':u,'arm':arm,'seed':seed,'protocol_sha256':sha(E/'configs/protocol.json'),'initial_hash':initial['hash']}
   for u in range(start+1,6001):
    for group in opt.param_groups:group['lr']=learning_rate(u,6000)
    k=0 if u<=5000 else 3;opt.zero_grad(set_to_none=True);accvals={}
    for acc in range(2):
     ids=schedule['indices'][u-1,acc];b={key:torch.stack([cached[int(i)][key]for i in ids]).cuda()for key in cached[0]}
     latentseed=int(schedule['latent_seeds'][u-1,acc]);torch.manual_seed(latentseed)
     with torch.autocast('cuda',dtype=torch.bfloat16):total,current,future,terms,enc=losses(m,b,k,w)
     assert torch.isfinite(total)
     route=backward_routed(m,total,current,enc,arm,divisor=2)
     for name,v in dict(terms,current=current,future=future,total=total).items():accvals[name]=accvals.get(name,0)+float(v.detach())/2
     del b,total,current,future,terms,enc
    en,rn=clip_groups(m);opt.step()
    if u==start+1 or u%100==0:
     row={'utc':now(),'size':SIZE,'pid':os.getpid(),'arm':arm,'seed':seed,'update':u,'target':6000,'k':k,'lr':learning_rate(u,6000),'seconds_this_segment':time.perf_counter()-begin,'E_preclip_norm':en,'R_preclip_norm':rn,'peak_GPU_bytes':torch.cuda.max_memory_allocated(),**accvals,**route};log.write(json.dumps(row)+'\n');log.flush();save(E/'results/progress.json',row);save(B/'results/progress.json',row);print('TRAIN',json.dumps(row),flush=True)
    if u%1000==0 or STOP:
     atomic_save(checkpoint(u),folder/'latest.pt')
     if u==6000:atomic_save({'model':m.state_dict(),'update':u,'seed':seed,'arm':arm,'initial_hash':initial['hash'],'protocol_sha256':sha(E/'configs/protocol.json')},folder/'final6000.pt')
    if STOP:
     attempts[-1]['status']='paused';attempts[-1]['saved_update']=u;save(E/'results/training_attempts.json',attempts);log.close();print('PAUSED',arm,seed,u,flush=True);return
   log.close();torch.cuda.synchronize();result={'utc':now(),'seed':seed,'arm':arm,'size':SIZE,'updates':6000,'resume_update':start,'elapsed_seconds_this_segment':time.perf_counter()-begin,'initial_hash':initial['hash'],'final_hash':th(m.state_dict()),'checkpoint_sha256':sha(folder/'final6000.pt'),'schedule_sha256':sha(E/f'configs/schedule_seed{seed}.npz'),'protocol_sha256':sha(E/'configs/protocol.json'),'peak_GPU_bytes':torch.cuda.max_memory_allocated(),'maxRSS_kB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'C_samples_read':0,'D_samples_read':0,'Cselection':False,'cached_loader_exact':True};save(resultfile,result);attempts[-1]['status']='complete';save(E/'results/training_attempts.json',attempts);print('FIT_COMPLETE',json.dumps(result),flush=True);del m,opt;torch.cuda.empty_cache()
 save(E/'configs/models_frozen.json',{'utc':now(),'before_dev_prediction':True,'sha256':{str(p):sha(p)for p in(E/'models').glob('*/final6000.pt')},'fits':4});print('ALL_FOUR_FITS_COMPLETE',flush=True)
 lock.close()
if __name__=='__main__':main()
