"""Actual one-skill-at-a-time execution. Evaluator truth is scored separately."""
from pathlib import Path
import json,time,subprocess,gzip,argparse,traceback,hashlib
import torch
from wm_symbolic.model_runtime import ModelRuntime
from wm_symbolic.search import DFA
from wm_symbolic.task_specs import parse_task_text
from wm_symbolic.checkpointing import PauseRequest

class ExecutorClient:
 def __init__(self,project,root,run):
  self.project=Path(project);self.run=Path(run);self.run.mkdir(parents=True,exist_ok=False);self.id=0;self.log=(self.run/'executor.log').open('w')
  self.process=subprocess.Popen(['bash','-c','source "$1"; shift; exec python "$@"','executor',str(self.project/'env.sh'),str(self.project/'scripts/executor_service.py'),'--root',str(root),'--run',str(run)],stdout=self.log,stderr=subprocess.STDOUT)
  self.ready=self.wait(self.run/'ready.json',120)
 def wait(self,path,timeout=120):
  start=time.monotonic()
  while not path.exists():
   if (self.run/'error.json').exists():raise RuntimeError((self.run/'error.json').read_text())
   if self.process.poll() is not None:raise RuntimeError('Executor exited; see '+str(self.run/'executor.log'))
   if time.monotonic()-start>timeout:raise TimeoutError(str(path))
   time.sleep(.02)
  return json.loads(path.read_text())
 def request(self,command,**args):
  path=self.run/'requests'/f'{self.id:03d}.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps({'command':command,**args}));tmp.replace(path)
  response=self.wait(self.run/'responses'/f'{self.id:03d}.json');self.id+=1
  if any(k not in {'public_frames','control_steps','hidden_feedback_returned','closed'} for k in response):raise ValueError('Undeclared executor feedback')
  return response
 def close(self):
  if self.process.poll() is None:
   try:self.request('shutdown');self.process.wait(timeout=30)
   except Exception:self.process.terminate();self.process.wait(timeout=30)
  self.log.close()

def run_task(project,checkpoint,task,root,run,calibration=None,ignore_ltl=False,seed=0,symbol_checkpoint=None,require_calibrated=False):
 p=Path(project);run=Path(run);parsed=parse_task_text(task['text'])
 if parsed['terminal_goals']!=task['terminal_goals'] or parsed['ltlf']!=task['ltlf']:raise ValueError('Task parser mismatch')
 pause=PauseRequest();torch.manual_seed(seed);client=ExecutorClient(p,root,run/'execution');runtime=None;begin=time.monotonic();trace=[];actions=0;steps=0;reobservations=0;queries=0;status='not_started'
 try:
  runtime=ModelRuntime(p,checkpoint,calibration,symbol_checkpoint);dfa=DFA(task['dfa']);progress=dfa.initial;progress_before_boundary=dfa.initial;consumed=False
  belief=runtime.update(client.ready['public_frames']);trace.append({'boundary':0,'observation':client.ready['public_frames'],'belief':belief,'actual_dfa_before':progress})
  for boundary in range(task['max_executed_skills']+1):
   if pause.requested:status='paused_by_user';break
   if (require_calibrated or task.get('root_id','').startswith('D_')) and not runtime.calibration.calibrated:status='uncalibrated_stop';break
   facts,unknown=runtime.facts(() if ignore_ltl else dfa.props)
   if unknown:
    if reobservations==0:
     response=client.request('reobserve');reobservations+=1;steps+=response['control_steps'];belief=runtime.update(response['public_frames'],reobserve=True)
     trace.append({'reobserve_at_boundary':boundary,'belief':belief,'cost_control_steps':response['control_steps'],'dfa_not_advanced_twice':True});facts,unknown=runtime.facts(() if ignore_ltl else dfa.props)
    if unknown:status='unknown_stop';break
   if not consumed:
    if not ignore_ltl:progress=dfa.advance(progress_before_boundary,facts)
    consumed=True
   if not ignore_ltl and progress not in dfa.distance:status='estimated_process_violation_stop';break
   goal=runtime.current_goal_score(task);trace.append({'boundary':boundary,'estimated_facts':facts,'actual_dfa_progress_from_observations':progress,'goal_score':goal})
   if goal>=runtime.calibration.threshold and (ignore_ltl or progress in dfa.accepting):
    status='confirmed_by_model' if runtime.calibration.calibrated else 'diagnostic_confirmation_uncalibrated';break
   if actions>=task['max_executed_skills']:status='skill_budget_exhausted';break
   runtime.save_state(run/f'planning_root_{actions:02d}.pt');planned=runtime.plan(task,dfa,progress,ignore_ltl);queries+=planned['queries']
   with gzip.open(run/f'plan_{actions:02d}.json.gz','wt') as f:json.dump(planned,f)
   if planned['status']!='candidate':
    if reobservations==0:
     response=client.request('reobserve');reobservations+=1;steps+=response['control_steps'];belief=runtime.update(response['public_frames'],reobserve=True)
     trace.append({'reobserve_at_boundary':boundary,'belief':belief,'cost_control_steps':response['control_steps'],'dfa_not_advanced_twice':True})
     # Replace the estimate of this boundary, without creating a second time step.
     facts,unknown=runtime.facts(() if ignore_ltl else dfa.props)
     if unknown:status='unknown_stop';break
     if not ignore_ltl:
      progress=dfa.advance(progress_before_boundary,facts)
      if progress not in dfa.distance:status='estimated_process_violation_stop';break
     goal=runtime.current_goal_score(task)
     if goal>=runtime.calibration.threshold and (ignore_ltl or progress in dfa.accepting):
      status='confirmed_by_model' if runtime.calibration.calibrated else 'diagnostic_confirmation_uncalibrated';break
     runtime.save_state(run/f'planning_root_{actions:02d}_after_reobserve.pt')
     planned=runtime.plan(task,dfa,progress,ignore_ltl);queries+=planned['queries']
     with gzip.open(run/f'plan_{actions:02d}_after_reobserve.json.gz','wt') as f:json.dump(planned,f)
    if planned['status']!='candidate':status='unsolved_within_search_budget';break
   selected=planned['plan'][0];response=client.request('skill',skill=selected);actions+=1;steps+=response['control_steps']
   belief=runtime.update(response['public_frames'],executed_skill=selected);progress_before_boundary=progress;consumed=False
   trace.append({'executed_skill':selected,'boundary':actions,'public_observation':response['public_frames'],'belief':belief,'planned_suffix_executed':False,'actual_dfa_before_observation_update':progress})
  result={'task_id':task['task_id'],'task':task,'status':status,'model_claimed_completion':status in ['confirmed_by_model','diagnostic_confirmation_uncalibrated'],
   'calibrated':runtime.calibration.calibrated,'physical_skills':actions,'physical_control_steps':steps,'reobservations':reobservations,'logical_successor_queries':queries,'RSSM_sample_evaluations':queries*8,
   'wall_seconds':time.monotonic()-begin,'checkpoint':str(checkpoint),'checkpoint_sha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
   'method':'full_without_process_constraints' if ignore_ltl else getattr(runtime,'method','full_RSSM_LTLf'),'symbol_checkpoint':str(symbol_checkpoint) if symbol_checkpoint else None,'trace':trace,'GT_feedback_used_by_model':False,'actual_success':'Pending independent evaluator; model confirmation is not ground-truth success'}
  (run/'runtime.json').write_text(json.dumps(result,indent=2))
  if status=='paused_by_user':(p/'runs/PAUSED_BY_USER.json').write_text(json.dumps({'status':'paused','stage':'closed_loop','run':str(run),'completed_skills':actions,'resume_requires_user_authorization':True},indent=2))
  return result
 except Exception:
  (run/'runtime_error.json').write_text(json.dumps({'error':traceback.format_exc(),'physical_skills':actions,'trace':trace},indent=2));raise
 finally:client.close()

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--checkpoint',required=True);a.add_argument('--tasks',required=True);a.add_argument('--task-index',type=int,required=True);a.add_argument('--root',required=True);a.add_argument('--run',required=True);a.add_argument('--calibration');a.add_argument('--symbol-checkpoint');a.add_argument('--require-calibrated',action='store_true');a.add_argument('--without-ltl',action='store_true');a.add_argument('--seed',type=int,default=0);args=a.parse_args()
 tasks=json.loads(Path(args.tasks).read_text());r=run_task('/home/robot/research/projects/wm_symbolic',args.checkpoint,tasks[args.task_index],args.root,args.run,args.calibration,args.without_ltl,args.seed,args.symbol_checkpoint,args.require_calibrated);print(json.dumps({k:v for k,v in r.items() if k not in ['trace','task']}))
