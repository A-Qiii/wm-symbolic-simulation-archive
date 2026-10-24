"""Predeclared C functional evaluation; all selected roots and failures retained."""
from pathlib import Path
import argparse,json,subprocess,datetime,signal,hashlib
from wm_symbolic.root_registry import active_root_directory
P=Path('/home/robot/research/projects/wm_symbolic');a=argparse.ArgumentParser();a.add_argument('--checkpoint',required=True);a.add_argument('--calibration',required=True);a.add_argument('--tag',required=True);a.add_argument('--seed',type=int,default=0);args=a.parse_args()
if (P/'runs/FORMAL_TRAINING_HOLD.json').exists():raise SystemExit('Large evaluation held for executor reliability review')
cal=json.loads(Path(args.calibration).read_text());assert cal['checkpoint_sha256']==hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
out=P/'runs'/('functional_'+args.tag);out.mkdir(exist_ok=True);status_path=out/'progress.json';current=None;pause=False
def save(**kw):
 d={'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw};tmp=status_path.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(status_path)
if not cal['calibration']['calibrated']:
 save(status='blocked_by_calibration',functional_acceptance_passed=False,reason='Independent C calibration did not support the prescribed confirmation error target; no uncalibrated physical success is counted as acceptance',calibration=args.calibration);raise SystemExit(2)
if (P/'runs/PAUSED_BY_USER.json').exists():raise SystemExit('Paused by user')
def stop(signum,frame):
 global pause
 pause=True
 (P/'runs/PAUSED_BY_USER.json').write_text(json.dumps({'status':'paused','stage':'functional_closed_loop','resume_requires_user_authorization':True,'last_task_saves_at_macro_boundary':True},indent=2))
 if current is not None and current.poll() is None:current.send_signal(signal.SIGUSR1)
for sig in [signal.SIGUSR1,signal.SIGINT,signal.SIGTERM]:signal.signal(sig,stop)
tasks_path=P/'configs/tasks/dev/task_specs.json';tasks=json.loads(tasks_path.read_text());chosen=list(enumerate(tasks))
assert len(chosen)==60 and len({t['root_id'] for _,t in chosen})==30
identity={'checkpoint':args.checkpoint,'calibration':args.calibration,'seed':args.seed,'task_ids':[t['task_id'] for _,t in chosen],'selection':'All30 C development roots, both task variants; all60 tasks evaluated by all three methods, failures retained.','tasks_sha256':hashlib.sha256(tasks_path.read_bytes()).hexdigest(),'methods':['reference','full','without_ltl'],'D_test_accessed':False}
idfile=out/'identity.json'
if idfile.exists():assert json.loads(idfile.read_text())==identity,'Refuse changed evaluation identity'
else:idfile.write_text(json.dumps(identity,ensure_ascii=False,indent=2))
records=[]
for index,task in chosen:
 root=active_root_directory(P,task['root_id'])
 for method in identity['methods']:
  taskdir=out/method/task['task_id'];taskdir.mkdir(parents=True,exist_ok=True);attempts=sorted(taskdir.glob('attempt_*'));complete=None
  for prior in attempts:
   if (prior/'independent_score.json').exists() and json.loads((prior/'runtime.json').read_text())['status']!='paused_by_user':complete=prior;break
  if complete is None:
   if pause or (P/'runs/PAUSED_BY_USER.json').exists():save(status='paused',completed=len(records));raise SystemExit(0)
   run=taskdir/f'attempt_{len(attempts):02d}'
   common=['--tasks',str(tasks_path),'--task-index',str(index),'--root',str(root),'--run',str(run)]
   if method=='reference':command=['scripts/run_reference.py',*common,'--references',str(P/'configs/tasks/dev/references_evaluator_only.json')]
   else:command=['scripts/closed_loop.py',*common,'--checkpoint',args.checkpoint,'--calibration',args.calibration,'--seed',str(args.seed),'--require-calibrated']+(['--without-ltl'] if method=='without_ltl' else [])
   with (taskdir/f'attempt_{len(attempts):02d}.log').open('w') as log:
    current=subprocess.Popen([str(P/'.envs/world_model/bin/python'),str(P/command[0]),*command[1:]],cwd=P,stdout=log,stderr=subprocess.STDOUT);save(status='running',method=method,task_id=task['task_id'],child_pid=current.pid,completed=len(records));code=current.wait()
   if code:save(status='blocked',method=method,task_id=task['task_id'],returncode=code);raise SystemExit(code)
   if pause or (P/'runs/PAUSED_BY_USER.json').exists():save(status='paused',completed=len(records));raise SystemExit(0)
   subprocess.run([str(P/'.envs/world_model/bin/python'),str(P/'scripts/score_closed_loop.py'),'--run',str(run),'--root',str(root)],check=True,stdout=subprocess.DEVNULL)
   complete=run
  score=json.loads((complete/'independent_score.json').read_text());runtime=json.loads((complete/'runtime.json').read_text());records.append({'method':method,'task_id':task['task_id'],'root_id':task['root_id'],'family':task['family'],'variant':task['variant'],'path':str(complete),'score':score,'runtime':{k:v for k,v in runtime.items() if k not in ['trace','task']},'actions':[x['executed_skill'] for x in runtime['trace'] if 'executed_skill' in x]})
  temp=out/'records.tmp';temp.write_text(json.dumps(records,indent=2));temp.replace(out/'records.json');print('FUNCTIONAL',method,task['task_id'],score['actual_success'],runtime['status'],flush=True)
full=[r for r in records if r['method']=='full'];reference={r['task_id']:r for r in records if r['method']=='reference'};without={r['task_id']:r for r in records if r['method']=='without_ltl'};valid_success=[r for r in full if r['score']['verified_success'] and 4<=r['score']['physical_skills']<=6]
family_roots={str(f):sorted({r['root_id'] for r in valid_success if r['family']==f}) for f in range(4)}
contrasts=[]
for root in sorted({r['root_id'] for r in valid_success}):
 pair=[r for r in valid_success if r['root_id']==root]
 if len(pair)==2 and pair[0]['family']==2 and pair[0]['actions']!=pair[1]['actions']:contrasts.append({'root_id':root,'tasks':[x['task_id'] for x in pair],'actions':[x['actions'] for x in pair]})
intermediate=[]
for r in valid_success:
 task=next(t for _,t in chosen if t['task_id']==r['task_id']);goal_colors={c for c in ['red','blue','pink'] if any(n.startswith(c+'_') for n in task['terminal_goals'])};used={c for c in ['red','blue','pink'] if any('_'+c+'_' in action for action in r['actions'])}
 if used-goal_colors:intermediate.append({'task_id':r['task_id'],'unmentioned_in_final_goal_blocks':sorted(used-goal_colors),'interpretation':'Successful plan explicitly operates a block absent from terminal goals; causal necessity is not established merely by occurrence'})
violation_avoided=[r['task_id'] for r in valid_success if without[r['task_id']]['score']['ltlf_accepted'] is False]
claims=[r for r in full if r['runtime']['model_claimed_completion']];false=sum(r['score']['false_completion_confirmation'] for r in claims);masked=sum(r['score']['completion_unverifiable_due_to_scoring_mask'] for r in claims);success=sum(r['score']['verified_success'] for r in full);ref_success=sum(r['score']['verified_success'] for r in reference.values());subset=[r for r in full if reference[r['task_id']]['score']['verified_success']]
passed=sum(len(v)>=3 for v in family_roots.values())>=3 and bool(contrasts) and bool(intermediate) and bool(violation_avoided) and bool(claims) and (false+masked)/len(claims)<=.05 and (ref_success==0 or success>=.6*ref_success)
result={'functional_acceptance_passed':passed,'family_success_roots_4_to_6_skills':family_roots,'same_root_different_constraint_successes':contrasts,'unmentioned_final_goal_intermediate_blocks':intermediate,'actual_violation_avoided_vs_without_LTL':violation_avoided,'full_denominator':len(full),'full_verified_successes':success,'reference_verified_successes':ref_success,'reference_executable_subset_denominator':len(subset),'candidate_successes_in_reference_subset':sum(r['score']['verified_success'] for r in subset),'completion_confirmations':len(claims),'known_false_confirmations':false,'scoring_unverifiable_confirmations':masked,'false_confirmation_upper_bound':(false+masked)/len(claims) if claims else None,'D_test_accessed':False,'all_failures_retained':True,'next':'Proceed to independent tests only after reviewing all functional criteria; otherwise preserve evidence and request a decision before changing the protocol'}
(out/'acceptance.json').write_text(json.dumps(result,indent=2));save(status='complete',functional_acceptance_passed=passed,completed=len(records));print('FUNCTIONAL_COMPLETE',json.dumps(result),flush=True)
