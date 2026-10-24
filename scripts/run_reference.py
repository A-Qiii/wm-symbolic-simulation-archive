"""Evaluator-only declared reference sequence, using the same frozen executor."""
from pathlib import Path
import argparse,json,time
from closed_loop import ExecutorClient
from wm_symbolic.checkpointing import PauseRequest
P=Path('/home/robot/research/projects/wm_symbolic');a=argparse.ArgumentParser();a.add_argument('--tasks',required=True);a.add_argument('--references',required=True);a.add_argument('--task-index',type=int,required=True);a.add_argument('--root',required=True);a.add_argument('--run',required=True);args=a.parse_args()
task=json.loads(Path(args.tasks).read_text())[args.task_index];reference=next(x for x in json.loads(Path(args.references).read_text()) if x['task_id']==task['task_id'])
run=Path(args.run);pause=PauseRequest();client=ExecutorClient(P,args.root,run/'execution');start=time.monotonic();trace=[];controls=0
try:
 for action in reference['actions']:
  if pause.requested:break
  response=client.request('skill',skill=action);controls+=response['control_steps'];trace.append({'executed_skill':action,'public_frames':response['public_frames']})
 result={'task_id':task['task_id'],'task':task,'method':'actual_reference_sequence_same_frozen_HULC','status':'paused_by_user' if pause.requested else 'reference_sequence_executed',
         'model_claimed_completion':False,'calibrated':False,'physical_skills':len(trace),'physical_control_steps':controls,
         'trace':trace,'wall_seconds':time.monotonic()-start,'reference_executable':'Pending independent scoring; sequence completion is not task success'}
 (run/'runtime.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['task','trace']}))
 if pause.requested:(P/'runs/PAUSED_BY_USER.json').write_text(json.dumps({'status':'paused','stage':'reference_execution','run':str(run),'completed_skills':len(trace),'resume_requires_user_authorization':True},indent=2))
finally:client.close()
