from common import *
import torch,fcntl,signal,copy
from wm_symbolic.world_model import WorldModel
from fixed_dataset import CoreEpisodes
from losses import losses,backward_routed,clip_groups
from original_losses import losses as original_losses
def dev_score(m,cache,w):
 torch.manual_seed(81831);scores=[]
 with torch.inference_mode():
  for off in range(0,len(cache),4):
   b={k:torch.stack([x[k]for x in cache[off:off+4]]).cuda()for k in cache[0]}
   with torch.autocast('cuda',dtype=torch.bfloat16):total,current,future,terms,enc,terminal=losses(m,b,3,w)
   scores.extend([float(total+terminal)]*len(b['skill']))
 return float(np.mean(scores))
def main():
 guard('train');assert not list((S/'models').glob('*.pt'));lock=(P/'runs/gpu.lock').open('r');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.monotonic();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in()).throw(TimeoutError('1800second frozen budget')));signal.alarm(1800);verify_inputs('train');torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('high');norm=read(CELL/'configs/normalization.json');data=CoreEpisodes(P,'train',norm);dev=CoreEpisodes(P,'validation',norm);cache=[data.get(i)for i in range(len(data.episodes))];vcache=[dev.get(i)for i in range(len(dev.episodes))];w=[torch.tensor(a,device='cuda')for a in norm['class_weights']];results=[];selection=[];updates=0
 assert set(data.root_ids)==set(read(S/'configs/protocol.json')['training_roots']);assert set(data.root_ids).isdisjoint(dev.root_ids)
 for seed in [0,1]:
  original=torch.load(CELL/f'models/joint_seed{seed}/latest.pt',map_location='cpu',weights_only=False);assert original['update']==6000;schedule=np.load(S/f'configs/schedule_seed{seed}.npz');basehash=None
  for arm in ARMS:
   m=WorldModel(skill_count=6,event_count=3).cuda();m.load_state_dict(original['model']);before=statehash(m);basehash=basehash or before;assert before==basehash
   final=torch.load(CELL/f'models/joint_seed{seed}/final6000.pt',map_location='cpu',weights_only=True);assert all(torch.equal(v.cpu(),final['model'][k])for k,v in m.state_dict().items());del final
   opt=torch.optim.AdamW(m.parameters(),lr=3e-5,betas=(.9,.999),weight_decay=1e-4);opt.load_state_dict(copy.deepcopy(original['optimizer']))
   for group in opt.param_groups:group['lr']=3e-5
   # Added measurement must leave the original loss, gradient and RNG unchanged.
   b={k:torch.stack([cache[i][k]for i in range(4)]).cuda()for k in cache[0]};torch.manual_seed(61811)
   with torch.autocast('cuda',dtype=torch.bfloat16):a=original_losses(m,b,3,w)
   rng=torch.cuda.get_rng_state();ga=torch.autograd.grad(a[0],m.posterior[0].weight)[0];orig_scalar=float(a[0]);del a
   torch.manual_seed(61811)
   with torch.autocast('cuda',dtype=torch.bfloat16):a=losses(m,b,3,w)
   gb=torch.autograd.grad(a[0],m.posterior[0].weight,retain_graph=True)[0];assert torch.equal(ga,gb)and torch.equal(rng,torch.cuda.get_rng_state())and orig_scalar==float(a[0]);assert torch.isfinite(a[-1]);del a,ga,gb,b
   m.train();begin=time.monotonic();torch.cuda.reset_peak_memory_stats();log=(S/f'logs/train_{arm}_seed{seed}.jsonl').open('w');curve=[];candidates=[]
   for step in range(1,751):
    opt.zero_grad(set_to_none=True);vals={}
    for acc in range(2):
     ids=schedule['indices'][step-1,acc];b={k:torch.stack([cache[int(i)][k]for i in ids]).cuda()for k in cache[0]};torch.manual_seed(int(schedule['latent_seeds'][step-1,acc]))
     with torch.autocast('cuda',dtype=torch.bfloat16):total,current,future,terms,enc,terminal=losses(m,b,3,w)
     objective=total+terminal if arm=='terminal'else total;curr=current+terminal if arm=='terminal'else current;assert torch.isfinite(objective);backward_routed(m,objective,curr,enc,'joint',2)
     for k,v in dict(terms,total=total,terminal=terminal,objective=objective).items():vals[k]=vals.get(k,0)+float(v.detach())/2
     del b,total,current,future,terms,enc,terminal,objective,curr
    en,rn=clip_groups(m);opt.step();updates+=1
    if step==1 or step%50==0:
     row={'utc':now(),'seed':seed,'arm':arm,'step':step,'seconds':time.monotonic()-begin,'E_norm':en,'R_norm':rn,'peak_GiB':torch.cuda.max_memory_allocated()/1024**3,**vals};log.write(json.dumps(row)+'\n');log.flush();curve.append(row);save(S/'results/progress.json',row);print('FIT',json.dumps(row),flush=True)
    if step in [250,500,750]:
     m.eval();score=dev_score(m,vcache,w);path=S/f'models/{arm}_seed{seed}_step{step}.pt';torch.save({'model':m.state_dict(),'arm':arm,'seed':seed,'continuation_update':step,'base_update':6000,'initial_hash':before,'protocol_sha256':sha(S/'configs/protocol.json')},path);candidates.append({'update':step,'dev_score':score,'path':str(path),'sha256':sha(path)});print('DEV',seed,arm,step,score,flush=True);m.train()
   log.close();chosen=min(candidates,key=lambda x:(x['dev_score'],x['update']));result={'utc':now(),'seed':seed,'arm':arm,'updates':750,'seconds':time.monotonic()-begin,'peak_GiB':torch.cuda.max_memory_allocated()/1024**3,'parameters':sum(p.numel()for p in m.parameters()),'initial_hash':before,'final_hash':statehash(m),'original_loss_scalar_gradient_rng_exact':True,'candidates':candidates,'selected':chosen,'curve':curve};save(S/f'results/train_{arm}_seed{seed}.json',result);results.append(result);selection.append({'seed':seed,'arm':arm,**chosen});del m,opt;torch.cuda.empty_cache()
  del original
 verify_inputs('train');save(S/'configs/models_frozen.json',{'utc':now(),'before_exploration_inference':True,'selection':selection,'fits':4,'updates':updates,'sha256':{c['path']:c['sha256']for r in results for c in r['candidates']}});save(S/'results/training_complete.json',{'utc':now(),'seconds':time.monotonic()-start,'fits':4,'continuation_updates':updates,'full6000step_retrains':0,'training_roots':data.root_ids,'selection_only_roots':dev.root_ids,'new_collection':0,'C_D_samples':0,'uploads':0});signal.alarm(0);print('TRAINING_COMPLETE',updates,time.monotonic()-start,flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:save(S/'results/TRAIN_FAILED.json',{'utc':now(),'error':repr(e),'no_automatic_retuning_or_budget_increase':True});raise
