"""Clean sequential reproduction; explicit --execute required for full work."""
import argparse,fcntl,json,os,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from paths import ROOT
from init_workspace import main as initialize
STAGES=[
 ('engineering','executor','engineering_gate.py',[],{'GATE_REV':'gate_06'},'engineering/gate_06/summary.json'),
 ('engineering-audit','executor','audit_engineering.py',[],{},'engineering/gate_06/audit.json'),
 ('collect-fit','executor','collect.py',[],{'COLLECTION_VERSION':'v1','COLLECT_MODE':'fit'},'data/v1/fit_complete.json'),
 ('cache-fit','world_model','cache_features.py',[],{'COLLECTION_VERSION':'v1','COLLECT_MODE':'fit'},'cache/v1/fit/complete.json'),
 ('smoke128','world_model','smoke128.py',[],{},'runs/v1/smoke128.json'),
 ('train-baselines','world_model','train.py',[],{},'configs/all_models_frozen.json'),
 ('collect-dev-cross','executor','v2_development_challenge.py',['collect'],{},'data/v2/development_cross/complete.json'),
 ('cache-dev-cross','world_model','v2_development_challenge.py',['cache'],{},'cache/v2/development_cross/complete.json'),
 ('train-v2','world_model','train_v2.py',[],{},'configs/v2_models_frozen.json'),
 ('train-v3','world_model','train_v3.py',[],{},'configs/v3_models_frozen.json'),
 ('collect-confirm','executor','confirm_v3_data.py',['collect'],{},'data/v3/confirmation/complete.json'),
 ('cache-confirm','world_model','confirm_v3_data.py',['cache'],{},'cache/v3/confirmation/complete.json'),
 ('evaluate','world_model','evaluate_v3.py',[],{},'reports/v3/confirmation/complete.json'),
 ('execute','world_model','live_v3.py',[],{},'reports/v3/live/complete.json'),
 ('analyze','world_model','analyze_v3.py',[],{},'reports/v3/analysis/provenance.json'),
 ('inspect-condition','world_model','inspect_v3_condition.py',[],{},'reports/v3/analysis/conditional_coefficients.json')]
def completion_valid(path):
 if not path.exists():return False
 q=json.loads(path.read_text())
 if 'all_passed'in q and not q['all_passed']:raise RuntimeError('Failed gate retained: '+str(path))
 if 'development_gate_passed'in q and not q['development_gate_passed']:raise RuntimeError('Development failure retained: '+str(path))
 return True
def main():
 p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--baseline-scope',choices=['transition_holdout','full'],default='transition_holdout');p.add_argument('--through',choices=[s[0]for s in STAGES]);a=p.parse_args()
 chosen=STAGES[:next(i+1 for i,s in enumerate(STAGES)if s[0]==a.through)]if a.through else STAGES
 if not a.execute:
  for name,env,script,args,extra,done in chosen:print(json.dumps({'stage':name,'python':f'.envs/{env}/bin/python','script':'src/'+script,'args':args,'environment':extra,'completion':done},ensure_ascii=False))
  return
 initialize();lock=(ROOT/'runs/reproduce.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  old=ROOT/'configs/all_models_frozen.json'
  if old.exists()and json.loads(old.read_text()).get('baseline_scope','full')!=a.baseline_scope:raise RuntimeError('Existing baseline scope differs; preserve this run and use a clean project copy')
  for name,env,script,args,extra,done in chosen:
   if completion_valid(ROOT/done):print('EXISTING_COMPLETED',name,flush=True);continue
   if name in ['collect-confirm','cache-confirm','evaluate','execute']and not completion_valid(ROOT/'configs/v3_models_frozen.json'):raise RuntimeError('v3 models not frozen')
   envfile=ROOT/('env.sh'if env=='executor'else'world_model_env.sh')
   runenv={**os.environ,**extra,'WM_BLOCKS_ROOT':str(ROOT),'WM_BLOCKS_BASELINE_SCOPE':a.baseline_scope}
   command=['bash','-c','source "$1"\nshift\nexec "$@"','bash',str(envfile),'python','-u',str(ROOT/'src'/script),*args]
   print('RUN',name,flush=True)
   with(ROOT/'logs'/('reproduce_'+name+'.log')).open('a')as f:subprocess.run(command,cwd=ROOT,env=runenv,stdout=f,stderr=subprocess.STDOUT,check=True)
   if not completion_valid(ROOT/done):raise RuntimeError('Stage returned without completion: '+name)
 finally:fcntl.flock(lock,fcntl.LOCK_UN)
if __name__=='__main__':main()
