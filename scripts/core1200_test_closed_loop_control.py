"""CPU control-flow regression: reobservation revises one boundary, not time."""
from pathlib import Path
from types import SimpleNamespace
import tempfile,json
import closed_loop as loop
P=Path('/home/robot/research/projects/wm_symbolic');seen=[]
class Client:
 def __init__(self,project,root,run):
  Path(run).mkdir(parents=True);self.ready={'public_frames':['initial']}
 def request(self,command,**kwargs):return {'public_frames':[command],'control_steps':4 if command=='reobserve' else 120}
 def close(self):pass
class Runtime:
 def __init__(self,*args):
  self.calibration=SimpleNamespace(threshold=.5,calibrated=False);self.observed='initial';self.plans=0
 def update(self,frames,**kwargs):self.observed=frames[0];return {'observed':self.observed}
 def facts(self,names):return ({'a':self.observed=='reobserve','b':self.observed=='skill'} if names else {}),None
 def current_goal_score(self,task):return float(self.observed=='skill')
 def save_state(self,path):Path(path).write_text('fixture only')
 def plan(self,task,dfa,progress,ignore):
  seen.append((progress,ignore));self.plans+=1
  return {'queries':25,'status':'candidate' if self.plans==2 or ignore else 'unsolved_within_budget','plan':['turn_on_led']}
class DFA:
 props=('a','b');initial=0;distance={0:2,1:1,2:0};accepting={2}
 def __init__(self,*args):pass
 def advance(self,state,facts):
  if state==0:return 1 if facts['a'] else 0
  if state==1:return 2 if facts['b'] else 1
  return 2
loop.ExecutorClient=Client;loop.ModelRuntime=Runtime;loop.DFA=DFA
loop.parse_task_text=lambda text:{'terminal_goals':{'b':True},'ltlf':'F(a & X(F(b)))'}
task={'task_id':'control_flow_fixture','text':'fixture','terminal_goals':{'b':True},'ltlf':'F(a & X(F(b)))','dfa':{},'max_executed_skills':8}
with tempfile.TemporaryDirectory(prefix='wm_control_test_',dir='/tmp') as directory:
 root=Path(directory);checkpoint=root/'fixture.pt';checkpoint.write_bytes(b'not a model')
 first=loop.run_task(P,checkpoint,task,root,root/'first')
 assert seen==[(0,False),(1,False)],seen
 assert first['physical_skills']==1 and first['reobservations']==1 and first['logical_successor_queries']==50
 assert first['model_claimed_completion'] and not first['calibrated']
 seen.clear();second=loop.run_task(P,checkpoint,task,root,root/'without_ltl',ignore_ltl=True)
 assert seen==[(0,True)] and second['physical_skills']==1 and second['reobservations']==0
 seen.clear();third=loop.run_task(P,checkpoint,dict(task,root_id='D_test_fixture'),root,root/'uncalibrated_D')
 assert third['status']=='uncalibrated_stop' and third['physical_skills']==0 and not third['model_claimed_completion'] and not seen
record={'passed':True,'reobservation_recomputes_same_boundary_from_pre_boundary_DFA':True,'imagined_progress_never_used_as_actual':True,
 'without_LTL_does_not_require_process_facts':True,'uncalibrated_confirmation_marked':True,'independent_D_confirmation_requires_calibration':True,'scope':'CPU control-flow fixtures; no physical or learned-model success claimed'}
(P/'runs/core1200_v1/closed_loop_control_regression.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
