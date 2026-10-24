"""Synthetic CPU test of the new loop's temporal bookkeeping, not an experiment."""
from pathlib import Path
from types import SimpleNamespace
import tempfile,json
import core1200_closed_loop as loop
P=Path('/home/robot/research/projects/wm_symbolic');seen=[]
class Client:
 def __init__(self,root,run):run.mkdir();self.ready={'public_frames':['initial']}
 def request(self,command,**kwargs):return {'public_frames':[command],'control_steps':4 if command=='reobserve' else 120}
 def close(self):pass
class Runtime:
 def __init__(self,*args):self.calibration=SimpleNamespace(threshold=.8,calibrated=False);self.obs='initial';self.plans=0
 def update(self,frames,**kwargs):self.obs=frames[0];return {'observed':self.obs}
 def facts(self,names):return ({'a':self.obs=='reobserve','b':self.obs=='skill'} if names else {}),None
 def current_goal_score(self,task):return float(self.obs=='skill')
 def save_state(self,path):path.write_text('synthetic fixture; no learned state')
 def plan(self,task,dfa,progress,ignore):
  seen.append((progress,ignore));self.plans+=1
  return {'queries':6,'status':'candidate' if self.plans==2 or ignore else 'unsolved_within_budget','plan':['open_drawer']}
class DFA:
 props=('a','b');initial=0;distance={0:2,1:1,2:0};accepting={2}
 def __init__(self,*args):pass
 def advance(self,state,facts):
  if state==0:return 1 if facts['a'] else 0
  if state==1:return 2 if facts['b'] else 1
  return 2
loop.Client=Client;loop.CoreRuntime=Runtime;loop.DFA=DFA
loop.score=lambda task,path:{'physical_task_success':True,'terminal_goal':True,'LTLf_satisfied':True,'scoring_valid':True,'setup_success':True}
task={'task_id':'synthetic_fixture','root_id':'synthetic_fixture','dfa':{},'terminal_goals':{'b':True}}
with tempfile.TemporaryDirectory(prefix='core_loop_unit_',dir='/tmp') as t:
 loop.R=Path(t)
 r=loop.run(task,'learned',[]);assert seen==[(0,False),(1,False)],seen
 assert r['physical_skills']==1 and r['reobservations']==1 and r['successor_queries']==12 and not r['completion_certified']
 seen.clear();r=loop.run(task,'goal_only',[]);assert seen==[(0,True)] and r['physical_skills']==1 and r['reobservations']==0
result={'passed':True,'root_revision_does_not_advance_macro_clock':True,'imagined_progress_not_used_as_actual':True,'without_LTL_has_same_goal_and_budget':True,'uncalibrated_confirmation_is_not_certified':True,'scope':'synthetic CPU control-flow fixture only; no real world-model or execution success asserted'}
(P/'runs/core1200_v1/new_loop_control_regression.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
