"""Single-GPU formal quota collection/cache. Safe pause prevents the next phase."""
from pathlib import Path
import subprocess,json,datetime,os,fcntl,signal
P=Path('/home/robot/research/projects/wm_symbolic');status=P/'runs/formal_data_pipeline.json'
if (P/'runs/PAUSED_BY_USER.json').exists():raise SystemExit('Paused by user')
lock=(P/'runs/formal_data_pipeline.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
current=None;pause_requested=False;phase='starting'
def save(**kwargs):
 record={'pid':os.getpid(),'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**kwargs};temporary=status.with_suffix('.tmp');temporary.write_text(json.dumps(record,indent=2));temporary.replace(status)
def request_pause(signum,frame):
 global pause_requested
 pause_requested=True
 if current is not None and current.poll() is None and (phase.startswith('collect_') or phase in ['cache_features','annotate_frozen_v2']):
  current.send_signal(signal.SIGUSR1)
for sig in [signal.SIGUSR1,signal.SIGINT,signal.SIGTERM]:signal.signal(sig,request_pause)
commands=[
 ('collect_train_4320','env.sh',['scripts/collect_dataset.py','--split','train','--roots','240','--branches','3']),
 ('collect_dev_360','env.sh',['scripts/collect_dataset.py','--split','dev','--roots','30','--branches','2']),
 ('collect_calibration_360','env.sh',['scripts/collect_dataset.py','--split','calibration','--roots','30','--branches','2']),
 ('collect_test_960','env.sh',['scripts/collect_dataset.py','--split','test','--roots','80','--branches','2']),
 ('annotate_frozen_v2','world_model_env.sh',['scripts/annotate_data_v2.py']),
 ('cache_features','world_model_env.sh',['scripts/cache_features.py'])]
for phase,environment,args in commands:
 if pause_requested or (P/'runs/PAUSED_BY_USER.json').exists():
  save(status='paused',next_phase=phase);raise SystemExit(0)
 with (P/'logs'/f'formal_{phase}.log').open('a') as log:
  current=subprocess.Popen(['bash','-c','source "$1"; shift; exec python "$@"','formal',str(P/environment),str(P/args[0]),*args[1:]],stdout=log,stderr=subprocess.STDOUT,cwd=P)
  save(status='running',step=phase,child_pid=current.pid);print('START',phase,current.pid,flush=True);code=current.wait()
 if pause_requested or (P/'runs/PAUSED_BY_USER.json').exists():
  if not (P/'runs/PAUSED_BY_USER.json').exists():(P/'runs/PAUSED_BY_USER.json').write_text(json.dumps({'status':'paused','stage':phase,'resume_requires_user_authorization':True},indent=2))
  save(status='paused',step=phase,child_returncode=code);raise SystemExit(0)
 if code:save(status='blocked',step=phase,returncode=code);raise SystemExit(code)
 print('COMPLETE',phase,flush=True)
m=json.loads((P/'data/data_manifest.json').read_text());counts={s:sum(r['split']==s for r in m['records']) for s in ['train','dev','calibration','test']}
assert counts=={'train':4320,'dev':360,'calibration':360,'test':960},counts
save(status='complete',step='formal_data_ready',split_counts=counts,next='Validate isolation and train the prescribed full models; no test-driven tuning')
