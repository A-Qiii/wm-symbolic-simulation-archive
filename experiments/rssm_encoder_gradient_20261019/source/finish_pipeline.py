"""Continue the existing run; never launches train.py or any main-model fit."""
from common import *
import subprocess,sys,time,fcntl
def main():
 lock=(E/'logs/finalization.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 last=0
 while not(E/'configs/models_frozen.json').exists():
  progress=read(E/'results/progress.json');attempts=read(E/'results/training_attempts.json')
  if attempts[-1]['status']=='paused':raise RuntimeError('Existing trainer paused; preserve checkpoint and inspect before recovery.')
  age=time.time()-(E/'results/progress.json').stat().st_mtime
  if age>900:raise RuntimeError('Existing trainer progress is stale; no duplicate fit launched.')
  if time.monotonic()-last>=60:
   print('WAITING_EXISTING_TRAINER',json.dumps({'complete':sum(a['status']=='complete'for a in attempts),'arm':progress['arm'],'seed':progress['seed'],'update':progress['update'],'utc':progress['utc']}),flush=True);last=time.monotonic()
  time.sleep(20)
 assert read(E/'configs/models_frozen.json')['fits']==4
 # Wait for original trainer to release the physical GPU lock after final manifest.
 gpu=(P/'runs/gpu.lock').open('r+')
 while True:
  try:fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(gpu,fcntl.LOCK_UN);break
  except BlockingIOError:time.sleep(1)
 gpu.close();print('FOUR_FITS_READY',flush=True)
 stages=[('evaluation',['evaluate.py']),('probe_train',['probes.py','train']),('probe_prediction',['probes.py','predict']),('score',['score.py']),('verify',['verify_results.py']),('report',['report.py'])]
 records=[]
 for stage,args in stages:
  for p,h in read(E/'configs/evaluation_frozen.json')['sha256'].items():assert sha(p)==h,p
  begin=time.time();save(E/'results/pipeline_status.json',{'utc':now(),'stage':stage,'status':'running','completed_stages':records});print('STAGE_START',stage,flush=True)
  with(E/'logs'/f'{stage}_console.log').open('a')as out:
   result=subprocess.run([sys.executable,str(E/'source'/args[0]),*args[1:]],stdout=out,stderr=subprocess.STDOUT)
  records.append({'stage':stage,'exit_code':result.returncode,'seconds':time.time()-begin})
  save(E/'results/pipeline_status.json',{'utc':now(),'stage':stage,'status':'failed'if result.returncode else'complete','completed_stages':records})
  print('STAGE_END',json.dumps(records[-1]),flush=True)
  if result.returncode:raise RuntimeError('Stage failed; inspect retained '+stage+'_console.log before any recovery.')
 print('ALL_EVALUATION_PROBES_SCORES_REPORT_READY',flush=True)
if __name__=='__main__':main()
