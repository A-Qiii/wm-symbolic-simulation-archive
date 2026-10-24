from common import *
import fcntl
def main():
    assert not (E/'results/existing_joint.json').exists()
    lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
    ds=dataset();tune={n:ds[n] for n in ['old_tune','new_dev']};cal={n:ds[n] for n in ['old_cal','new_dev']};out={};selected=[]
    for seed in range(3):
        choices=[]
        for tag,path in [('original',O/'models'/f'direct_predicates_seed{seed}.pt'),('prior_selected',A/'models'/f'augmented_seed{seed}_selected.pt'),('prior_final',A/'models'/f'augmented_seed{seed}_final.pt')]:
            name=f'{tag}_seed{seed}';m=loadmodel(path);preds={n:infer(m,ds[n]) for n in ['old_tune','old_cal','new_dev']}
            for n,p in preds.items():np.save(E/'results'/f'{name}_{n}.npy',p)
            rank,t=rank_predictions({n:preds[n] for n in tune},tune);c=calibrate_joint({n:preds[n] for n in cal},cal)
            out[name]={'path':str(path),'sha256':sha(path),'seed':seed,'rank':rank,'tune':t,'calibration':c,'qualified':t['gate']['passed'] and c['eligible']};choices.append((rank,name));del m
        name=min(choices)[1];selected.append(name)
    result={'created_utc':now(),'checkpoints':out,'joint_selected_names':selected,'qualified_selected_count':sum(out[n]['qualified'] for n in selected),'no_test_predictions_loaded':True,'limitation':'Only saved selected/final checkpoints are available; cannot rerank every historical update.'}
    save(E/'results/existing_joint.json',result)
    print(json.dumps({'selected':selected,'qualified_selected_count':result['qualified_selected_count'],'models':{n:{'tune_failures':v['tune']['gate']['failures'],'calibration_raw_failures':v['calibration']['raw_gate']['failures'],'qualified':v['qualified']} for n,v in out.items()}},ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
