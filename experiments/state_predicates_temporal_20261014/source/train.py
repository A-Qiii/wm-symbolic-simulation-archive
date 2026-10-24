from common import *
import fcntl,random
import torch.nn.functional as F
E=P/'experiments/state_predicates_temporal_20261014';J=P/'experiments/state_predicates_joint_20261014';ARMS=['current_only','short_history','long_history'];NAMES=['old_tune','old_cal','new_dev','fresh_tune','fresh_cal']
def build(arm):
 ds=dataset();meta=read(E/'data/metadata.json')
 with np.load(E/'data/temporal.npz')as z:arr={k:z[arm+'_'+k].copy()if k in ['objects','context','robot']else z[k].copy()for k in ['objects','context','robot','y','v']}
 ids=[i for i,r in enumerate(meta)if r['split']=='train'];assert len(ids)==36
 pool={k:np.concatenate([ds['old_train'][k],ds['new_train'][k],arr[k][ids]])for k in arr}
 for role in ['fresh_tune','fresh_cal']:
  ids=[i for i,r in enumerate(meta)if r['dev_role']==role];ds[role]={k:v[ids]for k,v in arr.items()};ds[role].update(roots=np.array([meta[i]['root_id']for i in ids]),meta=[meta[i]for i in ids])
 roots=set(ds['old_train']['roots'])|set(ds['new_train']['roots']);assert len(roots)==72
 for n in NAMES:assert not roots&set(ds[n]['roots'])
 return ds,pool
def main():
 assert not(E/'results/training.json').exists();protocol=read(E/'configs/protocol.json')
 for p,h in protocol['frozen_sha256'].items():assert sha(p)==h,p
 datasets={};pools={}
 for arm in ARMS:datasets[arm],pools[arm]=build(arm)
 for arm in ARMS[1:]:
  for k in ['y','v']:assert np.array_equal(pools[arm][k],pools[ARMS[0]][k])
  for k in ['objects','context','robot']:assert np.array_equal(pools[arm][k][:973],pools[ARMS[0]][k][:973])
 N=1009;assert all(len(p['y'])==N for p in pools.values());steps=600;weights=read(J/'configs/balanced_loss_loss.json')['weights'];batches={}
 for seed in range(3):
  rng=np.random.default_rng(seed);order=rng.permutation(N).tolist();cursor=0;ss=[]
  for _ in range(steps):
   while len(order)-cursor<60:order=order[cursor:]+rng.permutation(N).tolist();cursor=0
   ss.append(order[cursor:cursor+60]);cursor+=60
  batches[seed]=np.array(ss,np.int16);np.save(E/'logs'/f'matched_seed{seed}_batch_indices.npy',batches[seed])
 save(E/'configs/dataset_frozen.json',{'pool_per_arm':N,'same_labels_masks':True,'identical_historical_inputs':973,'temporal_train_windows':36,'metadata_sha256':sha(E/'data/metadata.json'),'arrays_sha256':sha(E/'data/temporal.npz')});(E/'configs/dataset_frozen.json').chmod(0o444)
 lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);torch.set_float32_matmul_precision('high');out={}
 for arm in ARMS:
  ds=datasets[arm];dat={k:torch.as_tensor(v,device='cuda')for k,v in pools[arm].items()};tune={n:ds[n]for n in ['old_tune','new_dev','fresh_tune']};cal={n:ds[n]for n in ['old_cal','new_dev','fresh_cal']};results=[]
  for seed in range(3):
   torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);m=PredicateReader().cuda();assert sum(p.numel()for p in m.parameters())==675395;opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.001);log=[];folder=E/'models'/arm/f'seed{seed}';folder.mkdir(parents=True,exist_ok=False)
   for update,batch in enumerate(batches[seed],1):
    ids=torch.as_tensor(batch.astype(np.int64),device='cuda');m.train();opt.zero_grad(set_to_none=True);o,g=m(**{k:dat[k][ids]for k in ['objects','context','robot']});l=torch.cat([o[:,:,0],o[:,:,1],g],-1);losses=F.binary_cross_entropy_with_logits(l,dat['y'][ids],reduction='none');parts=[]
    for name,cols in GROUPS.items():
     y=dat['y'][ids][:,cols];v=dat['v'][ids][:,cols];ww=y*weights[name]['positive']+(1-y)*weights[name]['negative'];parts.append((losses[:,cols]*v*ww).sum()/v.sum())
    loss=torch.stack(parts).mean();assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
    if update in protocol['training']['checkpoints']:
     preds={n:infer(m,d)for n,d in tune.items()};rank,sc=rank_predictions(preds,tune);cp=folder/f'update{update:04d}.pt';torch.save({'model':m.state_dict(),'seed':seed,'update':update,'arm':arm,'protocol_sha256':sha(E/'configs/protocol.json'),'rank':rank},cp);log.append({'update':update,'loss':float(loss.detach()),'rank':rank,'tune':sc,'checkpoint':str(cp)});save(E/'logs'/f'{arm}_seed{seed}.json',log)
   selected=min(log,key=lambda x:tuple(x['rank']));modes={}
   for mode,row in [('fixed600',log[-1]),('selected',selected)]:
    m=loadmodel(row['checkpoint']);pred={n:infer(m,ds[n])for n in NAMES}
    for n,v in pred.items():np.save(E/'results'/f'{arm}_seed{seed}_{mode}_{n}.npy',v)
    ca=calibrate_joint({n:pred[n]for n in cal},cal);modes[mode]={'update':row['update'],'checkpoint':row['checkpoint'],'checkpoint_sha256':sha(row['checkpoint']),'tune':row['tune'],'calibration':ca,'qualified':row['tune']['gate']['passed']and ca['eligible']}
   results.append({'seed':seed,'modes':modes,'updates':steps,'presentations':steps*60});save(E/'results'/f'{arm}_partial.json',results);print(json.dumps({'arm':arm,'seed':seed,'selected':selected['update'],'fixed_qualified':modes['fixed600']['qualified'],'selected_qualified':modes['selected']['qualified']},ensure_ascii=False),flush=True);del m,opt
  out[arm]=results;save(E/'results'/f'{arm}.json',results)
 out['scope']='No independent test; finite3x3x600update diagnostic completed. No further tuning.';save(E/'results/training.json',out);print('TEMPORAL_TRAINING_COMPLETE')
if __name__=='__main__':main()
