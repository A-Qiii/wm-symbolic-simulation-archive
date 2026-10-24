from common import *
import subprocess,sys,time

def main():
 assert not(B/'configs/all_models_frozen.json').exists(),'All8fits already complete; do not repeat'
 norms=[read(B/f'cells/roots{s}/configs/normalization.json')for s in [20,40]]
 assert norms[0]==norms[1] and len(norms[0]['fit_root_ids'])==20
 for seed in [0,1]:
  import torch
  a=[torch.load(B/f'cells/roots{s}/models/initial_seed{seed}.pt',map_location='cpu',weights_only=True)for s in [20,40]]
  assert a[0]['hash']==a[1]['hash'] and all(x['fresh_random_initialization']for x in a)
  q=[np.load(B/f'cells/roots{s}/configs/schedule_seed{seed}.npz')for s in [20,40]]
  assert np.array_equal(q[0]['latent_seeds'],q[1]['latent_seeds'])
  assert q[0]['indices'].shape==q[1]['indices'].shape==(6000,2,8)
  del a,q
 save(B/'results/launch.json',{'utc':now(),'pipeline_pid':os.getpid(),'command':sys.argv,'status':'launching','actual_first_update_is_separate_progress_receipt':True,'preprocessing_identical':True,'initialization_paired_acrosssizes':True})
 for size in [20,40]:
  env=os.environ.copy();env['RSSM_SCALE_ROOTS']=str(size)
  print('START_SIZE',size,'PIPELINE_PID',os.getpid(),now(),flush=True)
  with (B/'logs'/f'training_roots{size}_console.log').open('a')as log:
   subprocess.run([sys.executable,str(B/'source/train.py')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  print('COMPLETE_SIZE',size,now(),flush=True)
 files=list((B/'cells').glob('roots*/models/*/final6000.pt'));assert len(files)==8
 save(B/'configs/all_models_frozen.json',{'utc':now(),'all8_complete':True,'before_ABtest_orC_predictions':True,'sha256':{str(p):sha(p)for p in files}})
 save(B/'results/training_complete.json',{'utc':now(),'main_fits':8,'total_updates':48000,'no_D':True,'new_collection':0})
 print('ALL_EIGHT_MAIN_FITS_COMPLETE',flush=True)

if __name__=='__main__':main()
