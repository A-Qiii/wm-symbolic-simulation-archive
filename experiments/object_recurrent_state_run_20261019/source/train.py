from common import *
import torch,copy,fcntl,traceback
from models import Interface,ObjectGRU,GlobalGRU
from losses import loss_for_batch
from data import extract,load,batch
def newmodel(arm,seed):
 torch.manual_seed(2610207000+seed);interface=Interface();torch.manual_seed(2610207100+seed);return {'global':GlobalGRU,'object':ObjectGRU}[arm](interface)
def preflight():
 import losses_reference
 data=load('train',0,'cpu');b=batch(data,[0,120]);weights=[torch.tensor(x)for x in read(CELL/'configs/normalization.json')['class_weights']];checks=[]
 for arm in ARMS:
  m=newmodel(arm,0);n=copy.deepcopy(m);a,aa=losses_reference.loss_for_batch(m,b,weights);a.backward();v,vv=loss_for_batch(n,b,weights);v.backward();diff=max(float((p.grad-q.grad).abs().max())for p,q in zip(m.parameters(),n.parameters())if p.grad is not None);assert torch.allclose(a,v,atol=1e-5,rtol=1e-5)and diff<2e-5;checks.append({'arm':arm,'reference_loss':float(a.detach()),'batched_loss':float(v.detach()),'max_gradient_difference':diff})
 save(S/'results/preflight.json',{'utc':now(),'passed':True,'checks':checks,'optimizer_updates':0});print('PREFLIGHT',checks,flush=True)
def fit(arm,seed):
 budget(True);verify();data=load('train',seed);eps=read(S/'cache/train_metadata.json');ix={e['episode_id']:i for i,e in enumerate(eps)};order=read(S/'configs/episode_order.json');old=[ix[x]for x in order['old_episodes']];role=[[ix[x['root_id']+'_b1'],ix[x['root_id']+'_b2']]for x in order['new_roles']];schedule=dict(np.load(S/f'configs/schedule_seed{seed}.npz'));model=newmodel(arm,seed).cuda();initial_hash=statehash(model);interface_hash=statehash(model.interface);params=sum(p.numel()for p in model.parameters()if p.requires_grad);opt=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.999),weight_decay=1e-4);weights=[torch.tensor(x,device='cuda')for x in read(CELL/'configs/normalization.json')['class_weights']];start=time.time();log=[];update=0;torch.cuda.reset_peak_memory_stats()
 def checkpoint(tag):
  path=S/f'models/{arm}_seed{seed}_{tag}.pt';torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'update':update,'arm':arm,'seed':seed,'initial_state_sha256':initial_hash,'initial_interface_sha256':interface_hash,'parameter_count':params,'seconds':time.time()-start},path);return path
 try:
  for u in range(1500):
   budget(True);opt.zero_grad(set_to_none=True);termsum={}
   for micro in range(2):
    ids=[old[int(x)]for x in schedule['old_episode_index'][u,micro]]+[role[int(r)][int(b)-1]for r,b in schedule['new_role_branch'][u,micro]]
    with torch.autocast('cuda',dtype=torch.bfloat16):loss,terms=loss_for_batch(model,batch(data,ids),weights)
    assert torch.isfinite(loss),'nonfinite';(loss/2).backward()
    for k,v in {'total':loss,**terms}.items():termsum[k]=termsum.get(k,0)+float(v.detach())/2
   grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(grad);opt.step();update=u+1;log.append({'update':update,**termsum,'grad_norm':float(grad),'seconds':time.time()-start})
   if update%50==0:
    csvwrite(S/f'logs/{arm}_seed{seed}_train.csv',log);save(S/'results/progress.json',{'utc':now(),'arm':arm,'seed':seed,'update':update,'seconds':time.time()-start,'last_loss':termsum,'gpu_peak_bytes':torch.cuda.max_memory_allocated()});print('TRAIN',arm,seed,update,round(time.time()-start,1),round(termsum['total'],4),flush=True);assert torch.cuda.max_memory_allocated()<6*1024**3,'VRAM cap'
   if update in [500,1000,1500]:checkpoint(f'step{update}')
 except BaseException:
  checkpoint('partial');csvwrite(S/f'logs/{arm}_seed{seed}_train.csv',log);raise
 csvwrite(S/f'logs/{arm}_seed{seed}_train.csv',log);save(S/f'results/fit_{arm}_seed{seed}.json',{'utc':now(),'arm':arm,'seed':seed,'updates':update,'parameters':params,'initial_state_sha256':initial_hash,'initial_interface_sha256':interface_hash,'seconds':time.time()-start,'gpu_peak_bytes':torch.cuda.max_memory_allocated(),'final_sha256':sha(S/f'models/{arm}_seed{seed}_step1500.pt')});del model,opt,data;torch.cuda.empty_cache()
def main():
 guard();verify();lock=(P/'runs/gpu.lock').open();fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);torch.set_float32_matmul_precision('high');torch.use_deterministic_algorithms(True)
 try:
  for seed in [0,1]:
   for split in ['train','development']:extract(split,seed)
  preflight()
  for seed in [0,1]:
   for arm in ARMS:fit(arm,seed)
  budget(True);files=sorted((S/'models').glob('*step1500.pt'));assert len(files)==4;save(S/'configs/models_frozen.json',{'utc':now(),'checkpoint':1500,'sha256':{str(p):sha(p)for p in files},'no_selection':True});print('ALL_FOUR_FIXED_FITS_COMPLETE',flush=True)
 except BaseException as e:
  save(S/'results/STOP.json',{'utc':now(),'reason':str(e),'traceback':traceback.format_exc(),'no_retry':True});raise
if __name__=='__main__':main()
