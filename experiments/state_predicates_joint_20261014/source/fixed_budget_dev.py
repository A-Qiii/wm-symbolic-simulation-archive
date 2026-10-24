"""Post-freeze fixed-update diagnostic on development data only; never selects models."""
from common import *
import fcntl
def main():
    assert (E/'results/candidate_freeze.json').exists()and not(E/'results/fixed1500_dev_diagnostic.json').exists()
    lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);torch.set_float32_matmul_precision('high');ds=dataset();out={}
    for family in ['in_supervision','balanced_loss']:
        for seed in range(3):
            path=E/'models'/family/f'seed{seed}/update1500.pt';m=loadmodel(path);metrics={n:evaluate_predictions(infer(m,ds[n]),ds[n])for n in ['old_tune','old_cal','new_dev']};out[f'{family}_seed{seed}']={'update':1500,'checkpoint_sha256':sha(path),'metrics':metrics,'raw_gate':raw_gate(metrics)};del m
    save(E/'results/fixed1500_dev_diagnostic.json',{'created_utc':now(),'post_freeze_diagnostic_only':True,'no_test_input':True,'no_candidate_threshold_or_result_change':True,'models':out})
    print({k:r['raw_gate']['failures']for k,r in out.items()},flush=True)
if __name__=='__main__':main()
