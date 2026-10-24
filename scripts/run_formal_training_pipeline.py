"""One finite dependency chain for the authorized experiment; no scheduler."""
from pathlib import Path
import subprocess,json,datetime,os,fcntl,signal,time
P=Path('/home/robot/research/projects/wm_symbolic');status=P/'runs/formal_training_pipeline.json';pause=False;child=None;phase='waiting_for_formal_data'
if (P/'runs/FORMAL_TRAINING_HOLD.json').exists():raise SystemExit('Formal training held for executor reliability review')
lock=(P/'runs/formal_training_pipeline.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def save(**fields):
 record={'pid':os.getpid(),'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'phase':phase,**fields};temp=status.with_suffix('.tmp');temp.write_text(json.dumps(record,indent=2));temp.replace(status)
def request_pause(signum,frame):
 global pause
 pause=True
 if child is not None and child.poll() is None:child.send_signal(signal.SIGUSR1 if phase=='train_full_seed0_30000' else signal.SIGTERM)
for sig in [signal.SIGINT,signal.SIGTERM,signal.SIGUSR1]:signal.signal(sig,request_pause)
def paused():
 if pause or (P/'runs/PAUSED_BY_USER.json').exists():
  if not (P/'runs/PAUSED_BY_USER.json').exists():(P/'runs/PAUSED_BY_USER.json').write_text(json.dumps({'status':'paused','stage':phase,'resume_requires_user_authorization':True},indent=2))
  save(status='paused');return True
 return False
while True:
 if (P/'runs/FORMAL_TRAINING_HOLD.json').exists():save(status='held_for_executor_readiness_review');raise SystemExit(0)
 if paused():raise SystemExit(0)
 data=json.loads((P/'runs/formal_data_pipeline.json').read_text())
 if data['status']=='complete':break
 if data['status'] in ['blocked','paused']:save(status='blocked_by_data_pipeline',dependency=data);raise SystemExit(1)
 try:os.kill(data['pid'],0)
 except ProcessLookupError:save(status='blocked',reason='Data dependency stopped without completed status');raise SystemExit(1)
 save(status='waiting',dependency_phase=data.get('step'),dependency_pid=data['pid']);time.sleep(15)
commands=[('audit_formal_data',['scripts/audit_formal_data.py']),
 ('train_full_seed0_30000',['scripts/train_world_model.py','--mode','formal','--event-count','7','--seed','0']),
 ('analyze_full_seed0_C',['scripts/analyze_pilot.py','--checkpoint',str(P/'runs/wm_formal_seed0/best.pt'),'--output',str(P/'runs/formal_development_analysis_seed0')]),
 ('aggregate_full_seed0_C',['scripts/aggregate_prediction_analysis.py','--input',str(P/'runs/formal_development_analysis_seed0')]),
 ('render_formal_evidence',['scripts/render_formal_report.py'])]
for phase,args in commands:
 if paused():raise SystemExit(0)
 with (P/'logs'/f'formal_{phase}.log').open('a') as log:
  child=subprocess.Popen(['bash','-c','source "$1"; shift; exec python "$@"','formal',str(P/'world_model_env.sh'),str(P/args[0]),*args[1:]],cwd=P,stdout=log,stderr=subprocess.STDOUT)
  save(status='running',child_pid=child.pid);print('START',phase,child.pid,flush=True);code=child.wait()
 if paused():raise SystemExit(0)
 if code:save(status='blocked',returncode=code);raise SystemExit(code)
 print('COMPLETE',phase,flush=True)
save(status='complete',next='Review formal development evidence, resolve requested order semantics, then calibrate and run functional closed-loop tests. This chain never reads D test predictions.')
