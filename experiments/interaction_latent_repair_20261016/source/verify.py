"""Read-only delivery verifier; no environment, model inference, or training."""
from common import *
import subprocess
def main():
 f=read(E/'configs/delivery_manifest.json');total=0
 for p,h in f['sha256'].items():assert sha(p)==h,p;total+=Path(p).stat().st_size
 for fn in['training_frozen.json','before_D2_frozen.json']:
  for p,h in read(E/'configs'/fn)['sha256'].items():assert sha(p)==h,p
 a=read(E/'results/execution_audit.json');assert a['training_updates']==4000 and a['all_frozen_parameter_hashes_unchanged'];assert a['all_task_calls']<=234 and a['preparation_calls']<=24;assert a['offline']['planned_task_slots']==90 and a['closed_loop_episodes']==24;assert a['old_D_reexecutions']==0 and not a['D2_retraining_or_reselection']
 for seed in[0,1]:
  x=read(E/'results'/f'train_canonical_control_{seed}.json');y=read(E/'results'/f'train_canonical_matched_{seed}.json');assert x['initial']['initial_state_hash']==y['initial']['initial_state_hash'];assert x['updates']==y['updates']==1000
 old=subprocess.run([sys.executable,str(REVIEW/'source/verify.py')],check=True,capture_output=True,text=True)
 print(json.dumps({'passed':True,'new_files':len(f['sha256']),'new_bytes':total,'previous_review_and_original':old.stdout.strip(),'verification_only_no_training_or_physics':True}))
if __name__=='__main__':main()
