"""One-shot authorized local pipeline; serial GPU jobs, stop on any failure."""
from pathlib import Path
import subprocess,json,datetime,os,fcntl
P=Path('/home/robot/research/projects/wm_symbolic');status=P/'runs/pilot_pipeline.json'
if (P/'runs/PAUSED_BY_USER.json').exists():
 raise SystemExit('Paused by user; explicit authorization required.')
lock=(P/'runs/pilot_pipeline.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock.seek(0);lock.truncate();lock.write(str(os.getpid()));lock.flush()
def save(**kwargs):status.write_text(json.dumps({'pid':os.getpid(),'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**kwargs},indent=2))
probe=json.loads((P/'runs/wm_probe_seed0/metrics_sampling8_01000.json').read_text());checks=json.loads((P/'runs/wm_probe_seed0/input_isolation_checks.json').read_text())
if not checks['passed'] or probe['current_accuracy']<.95 or probe['prior_accuracy']<.95:
 save(status='blocked',reason='128-transition probe has not demonstrated overfit/input isolation',probe=probe);raise SystemExit(2)
commands=[('collect_train_720','env.sh',['scripts/collect_dataset.py','--split','train','--roots','60','--branches','2']),
 ('collect_dev_120','env.sh',['scripts/collect_dataset.py','--split','dev','--roots','10','--branches','2']),
 ('annotate_pilot','env.sh',['scripts/annotate_data.py']),
 ('cache_pilot','world_model_env.sh',['scripts/cache_features.py']),
 ('train_pilot_6000','world_model_env.sh',['scripts/train_world_model.py','--mode','pilot','--seed','0'])]
for name,env_script,args in commands:
 save(status='running',step=name);print('START',name,flush=True)
 with (P/'logs'/f'{name}.log').open('w') as log:
  result=subprocess.run(['bash','-c','source "$1"; shift; exec python "$@"','pipeline',str(P/env_script),str(P/args[0]),*args[1:]],stdout=log,stderr=subprocess.STDOUT,cwd=P)
 if result.returncode:
  save(status='blocked',step=name,returncode=result.returncode,log=str(P/'logs'/f'{name}.log'));raise SystemExit(result.returncode)
 print('COMPLETE',name,flush=True)
save(status='complete',step='pilot_training_complete',next='Inspect development curves and three traceable current/one-step/free-rollout cases before formal expansion')
