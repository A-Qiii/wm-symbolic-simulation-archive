"""Fixed diagnostic closed loop and same-controller reference; score afterward."""
from pathlib import Path
import argparse,json,subprocess,time,gzip,hashlib,signal,traceback
import torch
from wm_symbolic.core_runtime import CoreRuntime
from wm_symbolic.search import DFA
from wm_symbolic.state_labels import COLORS,LOCATIONS
P=Path('/home/robot/research/projects/wm_symbolic');R=P/'runs/core1200_v1';B=P/'data/core1200_v1';stop=False
def dump(path,x):
 tmp=path.with_suffix('.tmp.json');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False));tmp.replace(path)
class Client:
 def __init__(self,root,run):
  self.run=run;run.mkdir();self.index=0;self.log=(run/'executor.log').open('w')
  self.proc=subprocess.Popen(['bash','-c','source "$1"; shift; exec python "$@"','executor',str(P/'env.sh'),str(P/'scripts/core1200_executor_service.py'),'--root',str(B/'roots'/root),'--run',str(run)],stdout=self.log,stderr=subprocess.STDOUT)
  self.ready=self.wait(run/'ready.json')
 def wait(self,path):
  start=time.monotonic()
  while not path.exists():
   if (self.run/'error.json').exists():raise RuntimeError((self.run/'error.json').read_text())
   if self.proc.poll() is not None:raise RuntimeError('Executor exited: '+str(self.run/'executor.log'))
   if time.monotonic()-start>180:raise TimeoutError(str(path))
   time.sleep(.02)
  return json.loads(path.read_text())
 def request(self,command,**kwargs):
  dump(self.run/'requests'/f'{self.index:03d}.json',{'command':command,**kwargs});response=self.wait(self.run/'responses'/f'{self.index:03d}.json');self.index+=1
  assert set(response)<={'public_frames','control_steps','hidden_feedback_returned','closed'},'Private IPC leakage'
  return response
 def close(self):
  if self.proc.poll() is None:
   try:self.request('shutdown');self.proc.wait(timeout=30)
   except Exception:self.proc.terminate();self.proc.wait(timeout=30)
  self.log.close()
def facts(label,events):
 f={c+'_moved':bool(v) for c,v in zip(COLORS,events)}
 for j,c in enumerate(COLORS):
  for k,n in enumerate(LOCATIONS):f[c+'_'+n]=bool(label['categorical'][j]==k) if label['categorical_valid'][j] else None
 for j,names in [(3,['drawer_closed','drawer_middle','drawer_open']),(4,['slider_right','slider_middle','slider_left'])]:
  for k,n in enumerate(names):f[n]=bool(label['categorical'][j]==k) if label['categorical_valid'][j] else None
 f['hand_empty']=label.get('hand_empty_derived');return f
def score(task,run):
 private=run/'execution/evaluator_only';root=json.loads((private/'root.json').read_text());dfa=DFA(task['dfa']);initial=facts(root['label'],[0,0,0]);valid=all(initial.get(n) is not None for n in dfa.props);progress_before=dfa.initial
 state=dfa.advance(progress_before,initial) if valid else progress_before;last=initial;events=[0,0,0];trace=[{'boundary':0,'facts':initial,'dfa_state':state}];controls=0;statuses=[];boundary=0
 for req in sorted((run/'execution/requests').glob('*.json')):
  cmd=json.loads(req.read_text())
  if cmd['command']=='shutdown':continue
  skill=cmd.get('skill','reobserve');x=json.loads((private/f'{req.stem}_{skill}/summary.json').read_text());controls+=x['result']['controls']
  if cmd['command']=='skill':boundary+=1;progress_before=state;events=x['events'];statuses.append(x['result']['status'])
  else:events=[int(a or b) for a,b in zip(events,x['events'])]
  last=facts(x['after'],events);valid &= all(last.get(n) is not None for n in dfa.props)
  if valid:state=dfa.advance(progress_before,last)
  trace.append({'boundary':boundary,'request':cmd,'facts':last,'dfa_state':state,'result':x['result']})
 goal_valid=all(last.get(n) is not None for n in task['terminal_goals']);goal=goal_valid and all(last[n]==v for n,v in task['terminal_goals'].items());process=valid and state in dfa.accepting;setup=all(x['status']=='success' for x in root['setup'])
 result={'physical_task_success':bool(goal and process and setup),'terminal_goal':bool(goal),'LTLf_satisfied':bool(process),'scoring_valid':bool(valid and goal_valid),'setup_success':setup,'actual_skill_statuses':statuses,'physical_controls':controls,'physical_skills':boundary,'trace':trace,'source':'private actual physical labels and full-process moved events; never sent to main WM/planner'}
 dump(run/'score_private.json',result);return result
def run(task,method,reference,root_override=None,root_latent=None):
 suffix=method if root_override is None else 'root_label_replacement';dest=R/'evaluation/closed_loop'/suffix/task['task_id'];dest.parent.mkdir(parents=True,exist_ok=True)
 if (dest/'result.json').exists():return json.loads((dest/'result.json').read_text())
 if dest.exists():dest.rename(dest.with_name(dest.name+'_interrupted_'+str(int(time.time()))))
 dest.mkdir();torch.manual_seed(82101);client=Client(task['root_id'],dest/'execution');model=None;trace=[];status='not_started';calls=0;reobs=0;queries=0;begin=time.monotonic();dfa=DFA(task['dfa']);progress=dfa.initial;before=dfa.initial;override_used=False
 try:
  if method=='reference':
   for skill in reference:
    if stop:break
    response=client.request('skill',skill=skill);calls+=1;trace.append({'skill':skill,'control_steps':response['control_steps']})
   status='reference_sequence_executed'
  else:
   model=CoreRuntime(P,R/'training_main/best.pt',R/'evaluation/calibration/calibration.json');trace.append({'boundary':0,'belief':model.update(client.ready['public_frames'])})
   if root_latent is not None:
    from wm_symbolic.world_model import State
    saved=torch.load(root_latent,map_location='cpu',weights_only=False);fresh=model.state
    assert Path(saved['checkpoint'])==R/'training_main/best.pt'
    trace.append({'paired_latent_source':str(root_latent),'fresh_reencoding_max_h_difference':float((fresh.h-saved['h']).abs().max()),'fresh_reencoding_max_z_difference':float((fresh.z-saved['z']).abs().max()),'paired_experiment_uses_original_h_z_and_rng':True})
    model.state=State(saved['h'],saved['z']);torch.set_rng_state(saved['cpu_rng']);torch.cuda.set_rng_state(saved['cuda_rng'])
   ignore=method=='goal_only'
   while True:
    if stop:status='paused';break
    model.save_state(dest/f'planning_root_{calls:02d}_r{reobs}.pt')
    f,error=model.facts(() if ignore else dfa.props);goal=model.current_goal_score(task)
    if root_override is not None and calls==0:
     f={n:root_override[n] for n in dfa.props};error=None;goal=float(all(root_override.get(n)==v for n,v in task['terminal_goals'].items()));override_used=True
    if error:
     if reobs==0:
      response=client.request('reobserve');reobs+=1;trace.append({'boundary':calls,'reobserve':True,'belief':model.update(response['public_frames'],reobserve=True)});continue
     status='unknown_current_state';break
    progress=before if ignore else dfa.advance(before,f)
    trace.append({'boundary':calls,'facts':f,'goal_score':goal,'progress':progress,'root_label_override':override_used and calls==0})
    if not ignore and progress not in dfa.distance:status='estimated_process_violation';break
    if goal>=model.calibration.threshold and (ignore or progress in dfa.accepting):status='diagnostic_model_confirmation';break
    if calls>=8:status='skill_budget_exhausted';break
    planned=model.plan(task,dfa,progress,ignore);queries+=planned['queries']
    with gzip.open(dest/f'plan_{calls:02d}_{reobs}.json.gz','wt') as stream:json.dump(planned,stream)
    if planned['status']!='candidate':
     if reobs==0:
      response=client.request('reobserve');reobs+=1;trace.append({'boundary':calls,'reobserve':True,'belief':model.update(response['public_frames'],reobserve=True)});continue
     status='no_plan_within_budget';break
    skill=planned['plan'][0];response=client.request('skill',skill=skill);before=progress;calls+=1;trace.append({'boundary':calls,'skill':skill,'belief':model.update(response['public_frames'],executed_skill=skill),'planned_sequence':planned['plan']})
  dump(dest/'runtime.json',{'task':task,'method':method,'status':status,'trace':trace,'physical_skills':calls,'reobservations':reobs,'successor_queries':queries,'elapsed_seconds':time.monotonic()-begin,'diagnostic_only':method!='reference','completion_certified':False,'root_label_replacement':root_override is not None,'latent_state_not_replaced':True})
 finally:
  client.close()
  if model is not None:del model
  torch.cuda.empty_cache()
 actual=score(task,dest);result={'task_id':task['task_id'],'root_id':task['root_id'],'method':suffix,'status':status,'physical_skills':calls,'reobservations':reobs,'successor_queries':queries,'physical_task_success':actual['physical_task_success'],'terminal_goal':actual['terminal_goal'],'LTLf_satisfied':actual['LTLf_satisfied'],'scoring_valid':actual['scoring_valid'],'setup_success':actual['setup_success'],'model_claimed':status=='diagnostic_model_confirmation','false_confirmation':status=='diagnostic_model_confirmation' and not actual['physical_task_success'],'completion_certified':False,'elapsed_seconds':time.monotonic()-begin}
 result['model_claim_scope']='terminal_goal_only' if method=='goal_only' else 'terminal_goal_and_LTL'
 result['full_task_unmet_when_claiming']=result['false_confirmation']
 result['false_goal_confirmation']=bool(result['model_claimed'] and not actual['terminal_goal'])
 if method=='goal_only':result['false_confirmation']=result['false_goal_confirmation']
 dump(dest/'result.json',result);return result
def main():
 global stop
 a=argparse.ArgumentParser();a.add_argument('--methods',nargs='+',default=['learned','goal_only','reference']);a.add_argument('--limit',type=int,default=12);args=a.parse_args()
 def requested(sig,frame):
  global stop
  stop=True
 for sig in [signal.SIGUSR1,signal.SIGTERM,signal.SIGINT]:signal.signal(sig,requested)
 tasks=json.loads((R/'evaluation/tasks.json').read_text());refs={x['task_id']:x['reference'] for x in json.loads((R/'evaluation/references_evaluator_only.json').read_text())};results=[]
 for method in args.methods:
  for task in tasks[:args.limit]:
   result=run(task,method,refs[task['task_id']]);results.append(result);dump(R/'evaluation/closed_loop_progress.json',{'results':results,'complete':False});print(json.dumps(result),flush=True)
   if stop:return
 dump(R/'evaluation/closed_loop_progress.json',{'results':results,'complete':True})
if __name__=='__main__':main()
