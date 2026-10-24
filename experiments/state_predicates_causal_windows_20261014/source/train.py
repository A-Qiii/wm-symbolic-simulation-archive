from common import *
import fcntl,random,time
import torch.nn.functional as F
E=P/'experiments/state_predicates_causal_windows_20261014';J=P/'experiments/state_predicates_joint_20261014'
def build():
    ds=dataset();meta=read(E/'data/metadata.json')
    with np.load(E/'data/causal_train_dev.npz')as z:arr={k:z[k].copy()for k in z.files}
    rootlist=sorted(set(r['root_id']for r in meta if r['split']=='train'));eligible=[];pairs=[]
    for rid in rootlist:
        by={r['condition']:i for i,r in enumerate(meta)if r['root_id']==rid}
        if not all(c in by for c in ['hard_negative','ordinary_negative','held_positive']):continue
        eligible.append(rid);pairs.append(by)
        assert np.array_equal(arr['y'][by['hard_negative']],arr['y'][by['ordinary_negative']])and np.array_equal(arr['v'][by['hard_negative']],arr['v'][by['ordinary_negative']])
    assert len(eligible)>=2,'No complete matched training roots'
    for role in ['fresh_tune','fresh_cal']:
        ids=[i for i,r in enumerate(meta)if r['dev_role']==role];assert ids
        ds[role]={k:arr[k][ids]for k in ['objects','context','robot','y','v']};ds[role].update(roots=np.array([meta[i]['root_id']for i in ids]),meta=[meta[i]for i in ids])
    pools={}
    for arm in ['ordinary_negative','hard_negative']:
        ids=[i for pair in pairs for i in [pair[arm],pair['held_positive']]]
        pools[arm]={k:np.concatenate([ds['old_train'][k],ds['new_train'][k],arr[k][ids]])for k in ['objects','context','robot','y','v']}
    assert np.array_equal(pools['ordinary_negative']['y'],pools['hard_negative']['y']);assert np.array_equal(pools['ordinary_negative']['v'],pools['hard_negative']['v'])
    trainroots=set(ds['old_train']['roots'])|set(ds['new_train']['roots'])|set(eligible)
    for name in ['old_tune','old_cal','new_dev','fresh_tune','fresh_cal']:assert not trainroots&set(ds[name]['roots'])
    return ds,pools,eligible,meta
def main():
    assert not(E/'results/training.json').exists();ds,pools,eligible,meta=build();N=len(pools['hard_negative']['y']);weights=read(J/'configs/balanced_loss_loss.json')['weights'];save(E/'configs/training_dataset_frozen.json',{'complete_train_roots':eligible,'pool_size':N,'append_windows':N-973,'identical_target_masks_and_label_counts':True,'arms':list(pools),'old_data_unchanged':True,'new_dev_not_train':True,'metadata_sha256':sha(E/'data/metadata.json'),'arrays_sha256':sha(E/'data/causal_train_dev.npz')});(E/'configs/training_dataset_frozen.json').chmod(0o444)
    batches={}
    for seed in range(3):
        rng=np.random.default_rng(seed);order=rng.permutation(N).tolist();cursor=0;ss=[]
        for _ in range(1500):
            while len(order)-cursor<60:order=order[cursor:]+rng.permutation(N).tolist();cursor=0
            ss.append(order[cursor:cursor+60]);cursor+=60
        batches[seed]=np.array(ss,np.int16);np.save(E/'logs'/f'matched_seed{seed}_batch_indices.npy',batches[seed])
    lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);torch.set_float32_matmul_precision('high');tune={n:ds[n]for n in ['old_tune','new_dev','fresh_tune']};cal={n:ds[n]for n in ['old_cal','new_dev','fresh_cal']};out={}
    for arm,pool in pools.items():
        dat={k:torch.as_tensor(v,device='cuda')for k,v in pool.items()};results=[]
        for seed in range(3):
            torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);m=PredicateReader().cuda();assert sum(p.numel()for p in m.parameters())==675395;opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.001);best=None;chosen=None;log=[];folder=E/'models'/arm/f'seed{seed}';folder.mkdir(parents=True,exist_ok=False)
            for update,batch in enumerate(batches[seed],1):
                ids=torch.as_tensor(batch.astype(np.int64),device='cuda');m.train();opt.zero_grad(set_to_none=True);o,g=m(**{k:dat[k][ids]for k in ['objects','context','robot']});l=torch.cat([o[:,:,0],o[:,:,1],g],-1);losses=F.binary_cross_entropy_with_logits(l,dat['y'][ids],reduction='none');parts=[]
                for name,cols in GROUPS.items():
                    truth=dat['y'][ids][:,cols];mask=dat['v'][ids][:,cols];ww=truth*weights[name]['positive']+(1-truth)*weights[name]['negative'];parts.append((losses[:,cols]*mask*ww).sum()/mask.sum())
                loss=torch.stack(parts).mean();assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
                if update==1 or update%75==0:
                    preds={n:infer(m,d)for n,d in tune.items()};rank,sc=rank_predictions(preds,tune);cp=folder/f'update{update:04d}.pt';torch.save({'model':m.state_dict(),'seed':seed,'update':update,'arm':arm,'protocol_sha256':sha(E/'configs/protocol.json'),'rank':rank},cp);row={'update':update,'loss':float(loss.detach()),'rank':rank,'tune':sc,'checkpoint':str(cp)};log.append(row)
                    if best is None or rank<best:best=rank;chosen=row
                    save(E/'logs'/f'{arm}_seed{seed}.json',log)
            m=loadmodel(chosen['checkpoint']);preds={n:infer(m,ds[n])for n in ['old_tune','old_cal','new_dev','fresh_tune','fresh_cal']}
            for n,v in preds.items():np.save(E/'results'/f'{arm}_seed{seed}_{n}.npy',v)
            c=calibrate_joint({n:preds[n]for n in cal},cal);result={'arm':arm,'seed':seed,'selected_update':chosen['update'],'checkpoint_path':chosen['checkpoint'],'checkpoint_sha256':sha(chosen['checkpoint']),'tune':chosen['tune'],'calibration':c,'qualified':chosen['tune']['gate']['passed']and c['eligible'],'updates':1500,'presentations':90000};results.append(result);save(E/'results'/f'{arm}_partial.json',results);print(json.dumps({'arm':arm,'seed':seed,'selected':result['selected_update'],'qualified':result['qualified'],'tune_failures':result['tune']['gate']['failures'],'cal_failures':c['raw_gate']['failures']},ensure_ascii=False),flush=True);del m,opt
        out[arm]={'runs':results,'qualified_count':sum(r['qualified']for r in results)};save(E/'results'/f'{arm}.json',out[arm])
    out['no_test_inference']=True;out['export_rule_change_not_held_definition_change']=True;save(E/'results/training.json',out);print('MATCHED_TRAINING_COMPLETE')
if __name__=='__main__':main()
