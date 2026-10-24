from common import *
import mid_train,mid_data,torch,fcntl,inspect,traceback
# AMP index assignment requires an explicit dtype match. This only changes storage
# of the detached public-history prior; values are unchanged (BF16 -> FP32).
src=inspect.getsource(mid_data.warm_prior).replace('prior[ids]=state','prior[ids]=state.to(prior.dtype)')
namespace=dict(mid_data.__dict__);exec(compile(src,'warm_prior_dtype_assignment','exec'),namespace)
mid_train.warm_prior=namespace['warm_prior']
def main():
 guard();verify();lock=(P/'runs/gpu.lock').open();fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);torch.set_float32_matmul_precision('high');torch.use_deterministic_algorithms(True)
 save(S/'results/runtime_dtype_adjustment.json',{'utc':now(),'reason':'Autocast prior index assignment had BF16/FP32 mismatch before history_mix first optimizer update.','change':'Explicitly convert detached prior to destination FP32; BF16 values are exactly representable. No objective/schedule/architecture/input change.','completed_full_seed0_preserved':True,'original_exception_log':'logs/train_console.log','checkpoint_scores_seen':False,'history_mix_updates_before_error':0})
 try:
  mid_train.preflight()
  for seed in [0,1]:
   for arm in ARMS:
    done=S/f'results/fit_{arm}_seed{seed}.json'
    if done.exists():
     q=read(done);assert q['updates']==1500 and q['checkpoint_sha256']==sha(S/f'models/{arm}_seed{seed}_step1500.pt');print('KEEP_COMPLETED',arm,seed,flush=True);continue
    mid_train.fit(arm,seed)
  files=sorted((S/'models').glob('*step1500.pt'));assert len(files)==6;save(S/'configs/models_frozen.json',{'utc':now(),'sha256':{str(p):sha(p)for p in files},'no_selection':True});print('ALL_SIX_FITS_COMPLETE',flush=True)
 except BaseException as e:save(S/'results/STOP_runtime.json',{'utc':now(),'error':str(e),'traceback':traceback.format_exc()});raise
if __name__=='__main__':main()
