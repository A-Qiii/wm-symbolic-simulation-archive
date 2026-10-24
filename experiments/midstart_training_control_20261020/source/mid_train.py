from common import *
import torch,fcntl,copy,traceback
from models import ObjectAug,Interface
from mid_data import load,batch,crop,warm_prior
from losses import loss_for_batch
def newmodel(seed):
 torch.manual_seed(2610207000+seed);interface=Interface();torch.manual_seed(2610207100+seed);return ObjectAug(interface)
def preflight():
 import original_loss
 data=load('train',0,'cpu');b=batch(data,[0,120]);weights=[torch.tensor(x)for x in read(CELL/'configs/normalization.json')['class_weights']];m=newmodel(0);n=copy.deepcopy(m);a,_=original_loss.loss_for_batch(m,b,weights);a.backward();v,_=loss_for_batch(n,b,weights);v.backward();diff=max(float((p.grad-q.grad).abs().max())for p,q in zip(m.parameters(),n.parameters())if p.grad is not None);assert a==v and diff==0
 c=crop(b,[1,3]);n.zero_grad();l,terms=loss_for_batch(n,c,weights);l.backward();grad={name:float(sum(p.grad.abs().sum()for p in module.parameters()if p.grad is not None))for name,module in [('object_prior',n.object_prior),('object_observe',n.object_observe),('context_prior',n.context_prior),('context_observe',n.context_observe)]};assert all(x>0 for x in grad.values())
 q={k:x.clone()for k,x in b.items()};q['cat'].fill_(0);q['outcomes'].fill_(0)
 for j,st in enumerate([1,3]):
  for key in ['objects','context','availability']:q[key][j,st:]=0
  for key in ['verb','pointer','budget']:q[key][j,st:]=0
 assert torch.equal(warm_prior(n,b,[1,3]),warm_prior(n,q,[1,3]))
 save(S/'results/preflight.json',{'passed':True,'full_loss_exact':True,'gradient_difference':diff,'cold_crop_gradient_sums':grad,'warm_prior_invariant_to_labels_and_current_future_observations_actions':True,'optimizer_updates':0});print('PREFLIGHT_PASS',flush=True)
def fit(arm,seed):
 budget(True);data=load('train',seed);z=np.load(S/f'configs/schedule_seed{seed}.npz');m=newmodel(seed).cuda();init=statehash(m);opt=torch.optim.AdamW(m.parameters(),lr=3e-4,betas=(.9,.999),weight_decay=1e-4);weights=[torch.tensor(x,device='cuda')for x in read(CELL/'configs/normalization.json')['class_weights']];log=[];start=time.time();torch.cuda.reset_peak_memory_stats();update=0
 try:
  for u in range(1500):
   budget(True);opt.zero_grad(set_to_none=True);termsum={}
   for micro in range(2):
    ids=z['ids'][u,micro];starts=z['starts'][u,micro];b=batch(data,ids)
    with torch.autocast('cuda',dtype=torch.bfloat16):
     prior=warm_prior(m,b,starts)if arm=='history_mix'else None
     if arm!='full':b=crop(b,starts)
     loss,terms=loss_for_batch(m,b,weights,prior)
    assert torch.isfinite(loss);(loss/2).backward()
    for k,v in {'total':loss,**terms}.items():termsum[k]=termsum.get(k,0)+float(v.detach())/2
   grad=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);assert torch.isfinite(grad);opt.step();update=u+1;log.append({'update':update,**termsum,'grad_norm':float(grad),'seconds':time.time()-start})
   if update%100==0:
    csvwrite(S/f'logs/{arm}_seed{seed}.csv',log);save(S/'results/progress.json',{'arm':arm,'seed':seed,'update':update,'seconds':time.time()-start,'loss':termsum});print('TRAIN',arm,seed,update,round(time.time()-start,1),round(termsum['total'],4),flush=True)
  path=S/f'models/{arm}_seed{seed}_step1500.pt';torch.save({'model':m.state_dict(),'update':update,'arm':arm,'seed':seed,'initial_state_sha256':init,'parameter_count':sum(p.numel()for p in m.parameters()),'seconds':time.time()-start},path)
  parity=None
  if arm=='full':
   old=torch.load(OLD/f'models/geometry_seed{seed}_step1500.pt',map_location='cuda',weights_only=False)['model'];parity={'all_tensors_equal':all(torch.equal(v,old[k])for k,v in m.state_dict().items()),'max_difference':max(float((v-old[k]).abs().max())for k,v in m.state_dict().items())};assert parity['all_tensors_equal'],parity
  save(S/f'results/fit_{arm}_seed{seed}.json',{'utc':now(),'arm':arm,'seed':seed,'updates':update,'parameters':sum(p.numel()for p in m.parameters()),'initial_sha256':init,'seconds':time.time()-start,'max_gpu_bytes':torch.cuda.max_memory_allocated(),'full_original_parity':parity,'checkpoint_sha256':sha(path)});csvwrite(S/f'logs/{arm}_seed{seed}.csv',log)
 except BaseException:
  torch.save({'model':m.state_dict(),'update':update},S/f'models/{arm}_seed{seed}_partial.pt');raise
 del m,opt,data;torch.cuda.empty_cache()
def main():
 guard();verify();lock=(P/'runs/gpu.lock').open();fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);torch.set_float32_matmul_precision('high');torch.use_deterministic_algorithms(True)
 try:
  preflight()
  for seed in [0,1]:
   for arm in ARMS:fit(arm,seed)
  files=sorted((S/'models').glob('*step1500.pt'));assert len(files)==6;save(S/'configs/models_frozen.json',{'utc':now(),'sha256':{str(p):sha(p)for p in files},'no_selection':True});print('ALL_SIX_FITS_COMPLETE',flush=True)
 except BaseException as e:save(S/'results/STOP.json',{'utc':now(),'error':str(e),'traceback':traceback.format_exc()});raise
if __name__=='__main__':main()
