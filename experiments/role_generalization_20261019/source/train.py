from common import *
import torch,fcntl,copy
from model import ObservationStateWorldModel,shared_parameters,load_shared,add_new_optimizer_group
from old_dataset import CoreEpisodes
from dataset import NewEpisodes
from losses import losses,backward_routed,clip_groups
def batch(cache,ids):return {k:torch.stack([cache[int(i)][k]for i in ids]).cuda()for k in cache[0]}
def score(m,cache,w):
 torch.manual_seed(81831);values=[]
 with torch.inference_mode():
  for off in range(0,len(cache),3):
   b=batch(cache,range(off,min(off+3,len(cache))))
   with torch.autocast('cuda',dtype=torch.bfloat16):total,current,future,terms,enc,terminal=losses(m,b,3,w)
   values.extend([float(total+terminal)]*len(b['skill']))
 return float(np.mean(values))
def main():
 guard('train');verify_freeze();budget_check();assert not list((S/'models').glob('*.pt'));assert read(S/'results/audit_train_dev.json')['passed'];start=time.monotonic();lock=(P/'runs/gpu.lock').open();fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);torch.set_float32_matmul_precision('high');torch.use_deterministic_algorithms(True)
 norm=read(CELL/'configs/normalization.json');old=CoreEpisodes(P,'train',norm);new={a:NewEpisodes('train',a)for a in ARMS};dev=NewEpisodes('development');assert len(old.episodes)==120 and len(dev.episodes)==48
 for a in ARMS:assert len(new[a].episodes)==72 and new[a].transitions==432
 for a,b in zip(new['random'].episodes,new['balanced'].episodes):assert a['family_id']==b['family_id']and a['branch']==b['branch']
 oldcache=[old.get(i)for i in range(120)];newcache={a:[d.get(i)for i in range(72)]for a,d in new.items()};vcache=[dev.get(i)for i in range(48)];w=[torch.tensor(x,device='cuda')for x in norm['class_weights']];schedules={}
 for seed in [0,1]:
  rng=np.random.default_rng(261020000+seed);oldidx=rng.integers(120,size=(1000,2,4));cycles=[]
  while len(cycles)<8000:cycles.extend(rng.permutation(72).tolist())
  sch={'old':oldidx,'new':np.array(cycles[:8000]).reshape(1000,2,4),'latent':np.arange(2000).reshape(1000,2)+2610201000+seed*20000};assert np.ptp(np.bincount(sch['new'].ravel(),minlength=72))<=1;np.savez_compressed(S/f'configs/schedule_seed{seed}.npz',**sch);schedules[seed]=sch
 for n,p in [('data_manifest',S/'data/data_manifest.json'),('features_index',S/'data/features/index.json')]: (S/f'configs/train_input_{n}.json').write_bytes(p.read_bytes())
 refs=list((S/'source').glob('*.py'))+[S/f'configs/schedule_seed{s}.npz'for s in [0,1]]+[S/f'configs/train_input_{n}.json'for n in ['data_manifest','features_index']]+[S/'configs/protocol.json']
 refs +=[Path(v)for d in [old,*new.values(),dev]for v in d.cache.values()]+[Path(r['labels_path'])for d in [old,*new.values(),dev]for e in d.episodes for r in e['rows']]
 frozen={str(p):sha(p)for p in set(refs)};save(S/'configs/training_frozen.json',{'utc':now(),'before_updates':True,'old_train40':old.root_ids,'new_training_families':sorted({e['family_id']for e in new['random'].episodes}),'development_only_families':sorted({e['family_id']for e in dev.episodes}),'sha256':frozen,'new_unique_episodes_each':72,'new_transitions_each':432,'old_transitions_each':480})
 results=[];selections=[]
 for seed in [0,1]:
  cp=torch.load(CELL/f'models/joint_seed{seed}/latest.pt',map_location='cpu',weights_only=False);assert cp['update']==6000;sch=schedules[seed];initial=None
  for arm in ARMS:
   budget_check();m=ObservationStateWorldModel(skill_count=6,event_count=3).cuda();load_shared(m,cp['model']);before=statehash(m);initial=initial or before;assert before==initial
   opt=torch.optim.AdamW(shared_parameters(m),lr=.0001,betas=(.9,.999),weight_decay=.0001);opt.load_state_dict(copy.deepcopy(cp['optimizer']));assert len(opt.state)==112;add_new_optimizer_group(opt,m)
   for g in opt.param_groups:g['lr']=.0001
   m.train();begin=time.monotonic();torch.cuda.reset_peak_memory_stats();curve=[];candidates=[];log=(S/f'logs/train_{arm}_seed{seed}.jsonl').open('w')
   for step in range(1,1001):
    try:budget_check()
    except TimeoutError:
     torch.save({'model':m.state_dict(),'optimizer':opt.state_dict(),'step':step-1,'seed':seed,'arm':arm},S/f'models/SAFETY_STOP_{arm}_seed{seed}.pt');raise
    opt.zero_grad(set_to_none=True);values={}
    for acc in range(2):
     examples=[oldcache[int(i)]for i in sch['old'][step-1,acc]]+[newcache[arm][int(i)]for i in sch['new'][step-1,acc]];b={k:torch.stack([x[k]for x in examples]).cuda()for k in examples[0]};torch.manual_seed(int(sch['latent'][step-1,acc]))
     with torch.autocast('cuda',dtype=torch.bfloat16):total,current,future,terms,enc,terminal=losses(m,b,3,w)
     objective=total+terminal;assert torch.isfinite(objective);backward_routed(m,objective,current+terminal,enc,'joint',2)
     for k,v in dict(terms,terminal=terminal,objective=objective).items():values[k]=values.get(k,0)+float(v.detach())/2
     del b,total,current,future,terms,enc,terminal,objective
    en,rn=clip_groups(m);opt.step()
    if step==1 or step%50==0:
     row={'utc':now(),'seed':seed,'arm':arm,'step':step,'seconds':time.monotonic()-begin,'peak_GiB':torch.cuda.max_memory_allocated()/1024**3,'E_norm':en,'R_norm':rn,**values};curve.append(row);log.write(json.dumps(row)+'\n');log.flush();save(S/'results/training_progress.json',row);print('FIT',arm,seed,step,round(row['seconds'],1),values['objective'],flush=True)
    if step in [250,500,1000]:
     m.eval();devscore=score(m,vcache,w);path=S/f'models/{arm}_seed{seed}_step{step}.pt';torch.save({'model':m.state_dict(),'seed':seed,'arm':arm,'continuation_update':step,'initial_hash':before,'protocol_sha256':sha(S/'configs/protocol.json')},path);candidates.append({'update':step,'development_objective':devscore,'path':str(path),'sha256':sha(path),'eligible_selection':step in[500,1000]});m.train();print('DEV',arm,seed,step,devscore,flush=True)
   chosen=min([x for x in candidates if x['eligible_selection']],key=lambda x:(x['development_objective'],x['update']));rec={'utc':now(),'seed':seed,'arm':arm,'updates':1000,'initial_hash':before,'final_hash':statehash(m),'seconds':time.monotonic()-begin,'peak_GiB':torch.cuda.max_memory_allocated()/1024**3,'candidates':candidates,'selected':chosen,'curve':curve,'old_samples':8000,'new_samples':8000};save(S/f'results/train_{arm}_seed{seed}.json',rec);results.append(rec);selections.append({'seed':seed,'arm':arm,**chosen});log.close();del m,opt;torch.cuda.empty_cache()
  del cp
 for p,h in frozen.items():assert sha(p)==h,p
 save(S/'configs/models_frozen.json',{'utc':now(),'before_test_collection':True,'selection':selections,'fits':4,'updates':4000,'sha256':{c['path']:c['sha256']for r in results for c in r['candidates']}});save(S/'results/training_complete.json',{'utc':now(),'fits':4,'updates':4000,'seconds':time.monotonic()-start,'test_episodes_read':0,'CD_samples':0});print('TRAINING_COMPLETE',flush=True)
if __name__=='__main__':main()
