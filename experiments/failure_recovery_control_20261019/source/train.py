from common import *
import torch,fcntl,signal,copy
from wm_symbolic.world_model import WorldModel
from old_dataset import CoreEpisodes
from dataset import NewEpisodes
from losses import losses,backward_routed,clip_groups
def batch(cache,ids):
 return {k:torch.stack([cache[int(i)][k]for i in ids]).cuda()for k in cache[0]}
def score(m,cache,w):
 torch.manual_seed(81831);values=[]
 with torch.inference_mode():
  for off in range(0,len(cache),3):
   b=batch(cache,range(off,min(off+3,len(cache))))
   with torch.autocast('cuda',dtype=torch.bfloat16):total,current,future,terms,enc,terminal=losses(m,b,3,w)
   values.extend([float(total+terminal)]*len(b['skill']))
 return float(np.mean(values))
def main():
 guard('train');verify_freeze();assert not list((S/'models').glob('*.pt'));assert read(S/'results/audit_train_dev.json')['passed'];assert read(S/'results/families_train_dev.json')['passed'];assert read(S/'results/collection_train_dev_complete.json')['training_gate_passed'];start=time.monotonic();lock=(P/'runs/gpu.lock').open('r');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);signal.signal(signal.SIGALRM,lambda *_:(_ for _ in()).throw(TimeoutError('1800seconds paired training budget')));signal.alarm(1800);torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('high')
 norm=read(CELL/'configs/normalization.json');old=CoreEpisodes(P,'train',norm);new={arm:NewEpisodes('train',arm)for arm in ARMS};dev=NewEpisodes('development');assert len(old.episodes)==120 and len(dev.episodes)==18;assert all(len(d.episodes)==24 and d.transitions==144 for d in new.values())
 for a,b in zip(new['nominal'].episodes,new['recovery'].episodes):
  assert a['root_id']==b['root_id'];assert [r['skill']for r in a['rows']]==[r['skill']for r in b['rows']];assert a['branch']in [0,1]and b['branch']in [0,2]
 oldcache=[old.get(i)for i in range(len(old.episodes))];newcache={arm:[d.get(i)for i in range(len(d.episodes))]for arm,d in new.items()};vcache=[dev.get(i)for i in range(len(dev.episodes))];w=[torch.tensor(v,device='cuda')for v in norm['class_weights']];schedules={}
 for seed in [0,1]:
  rng=np.random.default_rng(261020000+seed);sch={'old':rng.integers(120,size=(1000,2,4)),'new':rng.integers(24,size=(1000,2,4)),'latent':np.arange(2000).reshape(1000,2)+2610201000+seed*20000};np.savez_compressed(S/f'configs/schedule_seed{seed}.npz',**sch);schedules[seed]=sch
 # Preserve exact training-era aggregate inputs: evaluation later appends only new held-out records.
 aggregates={};snapshots=[]
 for name,live in [('data_manifest',S/'data/data_manifest.json'),('features_index',S/'data/features/index.json')]:
  snapshot=S/f'configs/train_input_{name}.json';snapshot.write_bytes(live.read_bytes());snapshots.append(snapshot);aggregates[str(live)]={'snapshot':str(snapshot),'sha256':sha(live)}
 refs=[p for p in(S/'data').rglob('*')if p.is_file()and str(p)not in aggregates];refs+=snapshots+list((S/'source').glob('*.py'))+[S/f'configs/schedule_seed{s}.npz'for s in [0,1]]+[S/'configs/protocol.json'];refs += [Path(v)for v in old.cache.values()]+[Path(r['labels_path'])for e in old.episodes for r in e['rows']]+[P/'runs/core1200_v1/protocol.json',P/'data/core1200_v1/data_manifest.json',P/'data/core1200_v1/features/index.json'];frozen={str(p):sha(p)for p in refs};save(S/'configs/training_frozen.json',{'utc':now(),'before_first_update':True,'old_train_roots':old.root_ids,'new_train_roots':new['nominal'].root_ids,'new_development_only_roots':dev.root_ids,'new_transitions_each_arm':144,'old_transitions_each_arm':480,'sha256':frozen,'append_only_aggregate_inputs':aggregates})
 results=[];selections=[]
 for seed in [0,1]:
  cp=torch.load(CELL/f'models/joint_seed{seed}/latest.pt',map_location='cpu',weights_only=False);assert cp['update']==6000;schedule=schedules[seed];init_hash=None
  for arm in ARMS:
   m=WorldModel(skill_count=6,event_count=3).cuda();m.load_state_dict(cp['model']);before=statehash(m);init_hash=init_hash or before;assert before==init_hash;opt=torch.optim.AdamW(m.parameters(),lr=.0001,betas=(.9,.999),weight_decay=.0001);opt.load_state_dict(copy.deepcopy(cp['optimizer']))
   for g in opt.param_groups:g['lr']=.0001
   m.train();begin=time.monotonic();torch.cuda.reset_peak_memory_stats();curve=[];candidates=[];log=(S/f'logs/train_{arm}_seed{seed}.jsonl').open('w')
   for step in range(1,1001):
    opt.zero_grad(set_to_none=True);values={}
    for acc in range(2):
     examples=[oldcache[int(i)]for i in schedule['old'][step-1,acc]]+[newcache[arm][int(i)]for i in schedule['new'][step-1,acc]];b={k:torch.stack([x[k]for x in examples]).cuda()for k in examples[0]};torch.manual_seed(int(schedule['latent'][step-1,acc]))
     with torch.autocast('cuda',dtype=torch.bfloat16):total,current,future,terms,enc,terminal=losses(m,b,3,w)
     objective=total+terminal;assert torch.isfinite(objective);backward_routed(m,objective,current+terminal,enc,'joint',2)
     for k,v in dict(terms,terminal=terminal,objective=objective).items():values[k]=values.get(k,0)+float(v.detach())/2
     del b,total,current,future,terms,enc,terminal,objective
    en,rn=clip_groups(m);opt.step()
    if step==1 or step%50==0:
     row={'utc':now(),'seed':seed,'arm':arm,'step':step,'seconds':time.monotonic()-begin,'peak_GiB':torch.cuda.max_memory_allocated()/1024**3,'E_norm':en,'R_norm':rn,**values};curve.append(row);log.write(json.dumps(row)+'\n');log.flush();save(S/'results/training_progress.json',row);print('FIT',json.dumps(row),flush=True)
    if step in [500,1000]:
     m.eval();devscore=score(m,vcache,w);path=S/f'models/{arm}_seed{seed}_step{step}.pt';torch.save({'model':m.state_dict(),'seed':seed,'arm':arm,'continuation_update':step,'initial_hash':before,'protocol_sha256':sha(S/'configs/protocol.json')},path);candidates.append({'update':step,'development_objective':devscore,'path':str(path),'sha256':sha(path)});print('DEV',seed,arm,step,devscore,flush=True);m.train()
   chosen=min(candidates,key=lambda x:(x['development_objective'],x['update']));receipt={'utc':now(),'seed':seed,'arm':arm,'updates':1000,'initial_hash':before,'final_hash':statehash(m),'seconds':time.monotonic()-begin,'peak_GiB':torch.cuda.max_memory_allocated()/1024**3,'candidates':candidates,'selected':chosen,'curve':curve,'old_samples':8000,'new_samples':8000,'same_terminal_loss_both_arms':True};save(S/f'results/train_{arm}_seed{seed}.json',receipt);results.append(receipt);selections.append({'seed':seed,'arm':arm,**chosen});log.close();del m,opt;torch.cuda.empty_cache()
  del cp
 for p,h in frozen.items():assert sha(p)==h,p
 verify_freeze();save(S/'configs/models_frozen.json',{'utc':now(),'before_new_evaluation_collection':True,'selection':selections,'fits':4,'updates':4000,'sha256':{c['path']:c['sha256']for r in results for c in r['candidates']}});save(S/'results/training_complete.json',{'utc':now(),'fits':4,'updates':4000,'seconds':time.monotonic()-start,'new_data_transitions_each_arm':144,'old_data_transitions_each_arm':480,'new_evaluation_samples_read':0,'C_D_samples':0,'uploads':0});signal.alarm(0);print('TRAINING_COMPLETE',time.monotonic()-start,flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:save(S/'results/TRAIN_FAILED.json',{'utc':now(),'error':repr(e),'no_automatic_retuning_or_budget_increase':True});raise
