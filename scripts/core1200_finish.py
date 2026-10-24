"""Resume-safe serial continuation of the already authorized local study."""
from pathlib import Path
import json,subprocess,time,os,datetime,traceback,signal
P=Path('/home/robot/research/projects/wm_symbolic');R=P/'runs/core1200_v1';child=None;stop=False
def read(path):return json.loads(path.read_text())
def status(phase,**extra):
 value={'phase':phase,'pid':os.getpid(),'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra};tmp=R/'post_pipeline_progress.tmp.json';tmp.write_text(json.dumps(value,indent=2));tmp.replace(R/'post_pipeline_progress.json')
 h=read(P/'runs/continuation_handoff.json');h.update(stage='core1200_'+phase,post_pipeline_progress='runs/core1200_v1/post_pipeline_progress.json',next=phase);(P/'runs/continuation_handoff.json').write_text(json.dumps(h,indent=2));print(json.dumps(value),flush=True)
def request(sig,frame):
 global stop
 stop=True
 if child is not None and child.poll() is None:child.send_signal(signal.SIGUSR1)
def phase(name,environment,script,args=[]):
 global child
 if stop:raise RuntimeError('Pause requested; no next phase started')
 status(name);log=P/'logs'/f'core1200_post_{name}.log'
 with log.open('a') as stream:
  child=subprocess.Popen(['bash','-c','source "$1"; shift; exec python "$@"','core-phase',str(P/environment),str(P/'scripts'/script),*args],cwd=P,stdout=stream,stderr=subprocess.STDOUT)
  code=child.wait();child=None
 if code:raise RuntimeError(f'{name} exited{code}; see{log}')
def main():
 for sig in [signal.SIGUSR1,signal.SIGTERM,signal.SIGINT]:signal.signal(sig,request)
 status('waiting_for_main_training')
 while True:
  if stop:raise RuntimeError('Pause requested')
  try:
   if read(R/'training_main/result.json')['complete']:break
  except (FileNotFoundError,json.JSONDecodeError):pass
  if (R/'training_main/paused.json').exists():raise RuntimeError('Main training paused; checkpoint preserved')
  time.sleep(2)
 assert read(R/'training_main/result.json')['updates']==6000
 if not (R/'evaluation/model_test_freeze.json').exists():phase('calibration_and_freeze','world_model_env.sh','core1200_calibrate.py')
 if read(P/'data/core1200_v1/data_manifest.json')['real_transitions']<1200:phase('sealed_D_collection','env.sh','core1200_collect.py',['--limit','1200','--splits','test'])
 if not (R/'full1200_qa.json').exists():phase('full_data_QA','env.sh','core1200_qa.py',['--name','full1200_qa','--videos','--include-test'])
 assert read(R/'full1200_qa.json')['passed']
 idx=read(P/'data/core1200_v1/features/index.json')
 if sum(x['split']=='test' for x in idx['items'])<300:phase('D_feature_cache','world_model_env.sh','core1200_cache.py',['--splits','test'])
 if not (R/'evaluation/test_once/metrics.json').exists():phase('D_evaluation_once','world_model_env.sh','core1200_test_once.py')
 if not (R/'evaluation/closed_loop_progress.json').exists() or not read(R/'evaluation/closed_loop_progress.json')['complete']:phase('closed_loop_controls','world_model_env.sh','core1200_closed_loop.py')
 assert read(R/'evaluation/closed_loop_progress.json')['complete']
 if not (R/'evaluation/diagnosis.json').exists():phase('single_root_label_diagnosis','world_model_env.sh','core1200_diagnose.py')
 phase('final_report','world_model_env.sh','core1200_final_report.py')
 status('complete_pending_visual_review_and_final_video',result='runs/core1200_v1/RESULT.json',report='reports/core1200_final.html')
if __name__=='__main__':
 try:main()
 except Exception:
  status('paused' if stop else 'blocked',error=traceback.format_exc());raise
