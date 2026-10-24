from common import *
import subprocess,sys,time,fcntl

def main():
 lock=(B/'results/finish_watch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not(B/'results/ready_for_visual_review.json').exists()
 trainpid=read(B/'results/launch.json')['pipeline_pid']
 save(B/'results/finish_watch_launch.json',{'utc':now(),'pid':os.getpid(),'waiting_for_training_pid':trainpid,'no_parallel_GPU_work':True})
 print('WAITING_FOR_ALL8_TRAINING',trainpid,flush=True)
 while not(B/'configs/all_models_frozen.json').exists():
  try:os.kill(trainpid,0)
  except ProcessLookupError:raise RuntimeError('Training pipeline stopped before all8models frozen; inspect instead of silently retrying')
  time.sleep(30)
 for p,h in read(B/'configs/evaluation_source_frozen.json')['files'].items():assert sha(p)==h,p
 for script in ['finish_compute.py','score.py','report.py']:
  print('FINISH_STAGE_START',script,now(),flush=True)
  with (B/'logs'/('pipeline_'+script+'.log')).open('a')as log:
   subprocess.run([sys.executable,str(B/'source'/script)],stdout=log,stderr=subprocess.STDOUT,check=True)
  print('FINISH_STAGE_COMPLETE',script,now(),flush=True)
 save(B/'results/ready_for_visual_review.json',{'utc':now(),'training_evaluation_statistics_report_complete':True,'visual_review_and_finalseal_pending':True})
 print('READY_FOR_VISUAL_REVIEW_AND_FINAL_SEAL',flush=True)

if __name__=='__main__':main()
