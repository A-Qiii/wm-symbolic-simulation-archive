from study import *
from model import ObservationStateWorldModel,shared_parameters,load_shared,add_new_optimizer_group

def main():
 guard('train');start=time.monotonic();assert not(U/'configs/training_frozen.json').exists();old,new,dev,norm=prepared();assert len(new.episodes)==36
 lock=(P/'runs/gpu.lock').open('r');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);torch.set_float32_matmul_precision('high');torch.use_deterministic_algorithms(True)
 assert sha(U/'source/model.py')==sha(C/'source/model.py')
 protocol={'utc':now(),'single_factor':'Existing new-root branch composition changes from b1/b2 roughly1/2 each to b0/b1/b2 roughly1/3 each. Same root identity at every draw, unchanged old draw and latent RNG. Branches cyclically balanced within each root, counts differ at most1.','model':'Exact current residual model source; all original modules and W trainable as before. No loss/gate/regularizer/architecture change.','initialization':'Same original joint6000 model and AdamW112 shared state entries; same zero W and absent W moments.','old_data':'40roots120episodes480transitions unchanged','new_data':'Same12roots; unique episodes24->36, actual unique transitions144->216, all already collected/cached. Draw count unchanged8000 per fit; b1/b2 exposure each falls from about4000 to about2667.','training':'2seeds0/1,1000updates each; microbatch4old+4new,accumulate2; same lr1e-4,clip E/R1,BF16,deterministic,full k3+terminal.','selection':'Same dev6rootsall3branches; full objective500/1000 minimum,tie earlier; fixed1000 also reported. No dev training.','gates':{'train':'Both seeds h0held12/12, b1/b2 h2+future3 joint12/12 maintained; b0 whole observed h0-h6 trajectory correct root count AND b0 h2+future3 joint root count strictly improve over residual paired control.','retention':'Per seed, split and branch: any root with a true stable nonaction InDrawer prior-correct->posterior-wrong event is a forgetting root. Treatment forgetting root sets must be subsets of control sets, not just equal aggregate counts.','development':'Both seeds h0held and b1/b2 h2+future3 joint root counts do not regress versus residual paired control; b0 whole-trajectory root count does not regress; apply per-root retention condition. Report absolute low performance; these diagnostic gates are not guide acceptance.','historical_evaluation':'Prior6roots exploratory only, no selection/training.'},'resource':'Only2new1000-update fits; target20minutes, training1200secondcap. No new simulation,C/D,upload.'}
 save(U/'configs/protocol.json',protocol);schedules={};counts=[];rootorder=[new.episodes[i]['root_id']for i in range(0,36,3)]
 for seed in [0,1]:
  with np.load(C/f'configs/schedule_seed{seed}.npz')as z:original={k:z[k]for k in z.files}
  roots=original['new']//2;seen=np.zeros(12,np.int64);mapped=[]
  for rid in roots.reshape(-1):
   branch=(int(seen[rid])+int(rid)+seed)%3;seen[rid]+=1;mapped.append(int(rid)*3+branch)
  current={**original,'new':np.array(mapped,dtype=original['new'].dtype).reshape(original['new'].shape)};assert np.array_equal(current['new']//3,roots)
  np.savez_compressed(U/f'configs/schedule_seed{seed}.npz',**current);schedules[seed]=current
  for rid in range(12):
   oldc=np.bincount(original['new'].reshape(-1),minlength=24)[rid*2:rid*2+2];newc=np.bincount(current['new'].reshape(-1),minlength=36)[rid*3:rid*3+3];assert newc.max()-newc.min()<=1
   for branch in range(3):counts.append({'seed':seed,'root_id':rootorder[rid],'branch':branch,'control_samples':int(oldc[branch-1])if branch else 0,'balanced_samples':int(newc[branch])})
 csvwrite(U/'configs/exposure_counts.csv',counts)
 oldcache=[old.get(i)for i in range(len(old.episodes))];newcache={'balanced':[new.get(i)for i in range(len(new.episodes))]};vcache=[dev.get(i)for i in range(len(dev.episodes))];w=[torch.tensor(v,device='cuda')for v in norm['class_weights']];checks=[]
 for seed in [0,1]:
  receipt=read(C/f'results/train_state_update_seed{seed}.json');cp=torch.load(CELL/f'models/joint_seed{seed}/latest.pt',map_location='cpu',weights_only=False);m=ObservationStateWorldModel(skill_count=6,event_count=3).cuda();load_shared(m,cp['model']);assert statehash(m)==receipt['initial_hash'];m.eval()
  opt=torch.optim.AdamW(shared_parameters(m),lr=.0001,betas=(.9,.999),weight_decay=.0001);opt.load_state_dict(copy.deepcopy(cp['optimizer']));assert len(opt.state)==112;add_new_optimizer_group(opt,m);assert m.observation_to_h.weight not in opt.state
  for candidate in receipt['candidates']:
   assert sha(candidate['path'])==candidate['sha256'];m.load_state_dict(torch.load(candidate['path'],map_location='cuda',weights_only=False)['model']);score=legacy_train.score(m,vcache,w);assert score==candidate['development_objective'];checks.append({'seed':seed,'update':candidate['update'],'stored':candidate['development_objective'],'recomputed':score})
  del m,opt,cp;torch.cuda.empty_cache()
 save(U/'results/control_reuse_audit.json',{'utc':now(),'reuse_allowed':True,'initial_full_state_exact':True,'optimizer_shared112_restored_newW_moments_absent':True,'same_model_loss_loop_budget_dev_rule':True,'checks':checks})
 text=(C/'source/training_loop.py').read_text();(U/'source/training_loop.py').write_text(text)
 refs=dict(read(C/'configs/training_frozen.json')['sha256']);refs.update(read(C/'configs/models_frozen.json')['sha256']);refs.update({str(p):sha(p)for p in(C/'source').glob('*.py')});refs.update({str(p):sha(p)for p in(C/'results').glob('train_state_update_seed*.json')});refs.update({str(p):sha(p)for p in(U/'source').glob('*.py')});refs.update({str(p):sha(p)for p in(U/'configs').glob('*')if p.is_file()})
 for p,h in refs.items():assert sha(p)==h,p
 save(U/'configs/training_frozen.json',{'utc':now(),'before_first_update':True,'sha256':refs});signal.signal(signal.SIGALRM,lambda *_:(_ for _ in()).throw(TimeoutError('1200second cap')));signal.alarm(1200)
 env=dict(globals(),S=U,ARMS=['balanced'],oldcache=oldcache,newcache=newcache,vcache=vcache,w=w,schedules=schedules,results=[],selections=[],score=legacy_train.score);exec(compile(text,str(U/'source/training_loop.py'),'exec'),env)
 for p,h in refs.items():assert sha(p)==h,p
 save(U/'configs/models_frozen.json',{'utc':now(),'selection':env['selections'],'fits':2,'updates':2000,'control_reused':True,'sha256':{c['path']:c['sha256']for r in env['results']for c in r['candidates']}});save(U/'results/training_complete.json',{'utc':now(),'fits':2,'updates':2000,'seconds':time.monotonic()-start,'new_collection':0,'C_D_samples':0});signal.alarm(0);print('TRAINING_COMPLETE',flush=True)

if __name__=='__main__':main()
