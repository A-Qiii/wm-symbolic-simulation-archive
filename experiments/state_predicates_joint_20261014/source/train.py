from common import *
import argparse,fcntl,time,random,shutil
import torch.nn.functional as F

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--variant',choices=['in_supervision','balanced_loss'],required=True);variant=ap.parse_args().variant
    assert not (E/'results'/f'{variant}.json').exists()
    if variant=='balanced_loss':assert read(E/'results/in_supervision.json')['qualified_count']<2
    ds=dataset();audit=[]
    for r in ds['new_train']['meta']:
        lab=read(r['labels_private']);j=r['target_index'];cats=[f['labels']['categorical'][j] for f in lab['frames']];valid=[f['labels']['categorical_valid'][j] for f in lab['frames']]
        assert all(valid) and all(c!=3 for c in cats)
        audit.append({'sample_id':r['sample_id'],'source':r['labels_private'],'sha256':sha(r['labels_private']),'four_frame_categories':cats,'all_valid':True,'target_in':False})
    save(E/'configs'/f'{variant}_decision.json',{'created_utc':now(),'variant':variant,'trigger':'Joint dev exposes new in FP and an existing C in failure; target-in labels absent from earlier supplemental training.' if variant=='in_supervision' else 'First single-mask change did not yield2/3 qualified seeds; use predeclared class/predicate-balanced loss only.','same_architecture':True,'supplementary_target_in_audit':audit,'test_results_not_used':True})
    (E/'configs'/f'{variant}_decision.json').chmod(0o444)
    lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
    dat={k:torch.tensor(np.concatenate([ds[n][k] for n in ['old_train','new_train']]),device='cuda') for k in ['objects','context','robot']}
    yy=np.concatenate([ds[n]['y'] for n in ['old_train','new_train']]);vv=np.concatenate([ds[n]['v'] for n in ['old_train','new_train']]);y=torch.tensor(yy,device='cuda');v=torch.tensor(vv,device='cuda')
    assert vv[:900].sum()==6300 and vv[900:].sum()==219
    tune={n:ds[n] for n in ['old_tune','new_dev']};cal={n:ds[n] for n in ['old_cal','new_dev']};weights={}
    for g,cols in GROUPS.items():
        pos=float((yy[:,cols]*vv[:,cols]).sum());neg=float(((1-yy[:,cols])*vv[:,cols]).sum());weights[g]={'positive':(pos+neg)/(2*pos),'negative':(pos+neg)/(2*neg),'n_positive':pos,'n_negative':neg}
    save(E/'configs'/f'{variant}_loss.json',{'variant':variant,'weights':weights if variant=='balanced_loss' else 'unweighted_per_valid_fact','balanced_formula':'mean_g [ sum_valid BCE * (N_g/(2Npos_g) if y=1 else N_g/(2Nneg_g)) / count_valid_g ]' if variant=='balanced_loss' else None})
    out=[]
    for seed in range(3):
        torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);rng=np.random.default_rng(seed);m=PredicateReader().cuda();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.001)
        order=rng.permutation(973).tolist();cursor=0;best=None;selected=None;log=[];batches=[];start=time.monotonic();folder=E/'models'/variant/f'seed{seed}';folder.mkdir(parents=True,exist_ok=False)
        for update in range(1,1501):
            while len(order)-cursor<60:order=order[cursor:]+rng.permutation(973).tolist();cursor=0
            indices=np.array(order[cursor:cursor+60]);cursor+=60;batches.append(indices);ids=torch.tensor(indices,device='cuda')
            m.train();opt.zero_grad(set_to_none=True);o,d=m(dat['objects'][ids],dat['context'][ids],dat['robot'][ids]);logits=torch.cat([o[:,:,0],o[:,:,1],d],-1);losses=F.binary_cross_entropy_with_logits(logits,y[ids],reduction='none')
            if variant=='in_supervision':loss=(losses*v[ids]).sum()/v[ids].sum()
            else:
                parts=[]
                for g,cols in GROUPS.items():
                    mask=v[ids][:,cols];truth=y[ids][:,cols];w=truth*weights[g]['positive']+(1-truth)*weights[g]['negative'];parts.append((losses[:,cols]*mask*w).sum()/mask.sum())
                loss=torch.stack(parts).mean()
            assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
            if update==1 or update%75==0:
                preds={n:infer(m,ds[n]) for n in tune};rank,score=rank_predictions(preds,tune);cp=folder/f'update{update:04d}.pt';torch.save({'model':m.state_dict(),'seed':seed,'update':update,'variant':variant,'protocol_sha256':sha(E/'configs/protocol.json'),'rank':rank},cp)
                row={'update':update,'loss':float(loss.detach()),'rank':rank,'tune':score,'checkpoint':str(cp)};log.append(row)
                if best is None or rank<best:best=rank;selected=row
                save(E/'logs'/f'{variant}_seed{seed}.json',log)
        np.save(E/'logs'/f'{variant}_seed{seed}_batch_indices.npy',np.array(batches,np.int16));path=Path(selected['checkpoint']);m=loadmodel(path)
        preds={n:infer(m,ds[n]) for n in ['old_tune','old_cal','new_dev']}
        for n,p in preds.items():np.save(E/'results'/f'{variant}_seed{seed}_{n}.npy',p)
        c=calibrate_joint({n:preds[n] for n in cal},cal)
        r={'seed':seed,'variant':variant,'checkpoint_path':str(path),'checkpoint_sha256':sha(path),'selected_update':selected['update'],'total_updates':1500,'presentations':90000,'tune':selected['tune'],'calibration':c,'qualified':selected['tune']['gate']['passed'] and c['eligible'],'seconds':time.monotonic()-start};out.append(r);save(E/'results'/f'{variant}_partial.json',out)
        print(json.dumps({'variant':variant,'seed':seed,'selected_update':r['selected_update'],'qualified':r['qualified'],'tune_failures':r['tune']['gate']['failures'],'cal_raw_failures':c['raw_gate']['failures']},ensure_ascii=False),flush=True)
        del m,opt
    save(E/'results'/f'{variant}.json',{'created_utc':now(),'variant':variant,'seeds':out,'qualified_count':sum(r['qualified'] for r in out),'no_test_loaded':True,'training_inputs_same_except_explicit_valid_mask':True,'source_sha256':{str(p):sha(p) for p in (E/'source').glob('*.py')}})
if __name__=='__main__':main()
