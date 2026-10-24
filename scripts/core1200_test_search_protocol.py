"""Synthetic finite-graph unit test: Dijkstra and whole-sample acceptance."""
from pathlib import Path
import heapq,json
import torch
from torch import nn
from wm_symbolic.world_model import State
from wm_symbolic.search import LearnedSearch,DFA
from wm_symbolic.symbol_decoder import Calibration
P=Path('/home/robot/research/projects/wm_symbolic');torch.set_num_threads(2)
class FixtureModel(nn.Module):
 transition_source='synthetic_unit_fixture_not_experimental_model'
 event_count=7
 def action(self,skills,arguments,budgets):return skills.float()[:,None]
 def imagine(self,state,action,deterministic=False):
  # Action0 fails to establish the goal in sample7. Action2 violates a constraint there.
  value=action[:,0]+1;return State(torch.stack([value,state.h[:,1]],-1),state.z),None
 def symbols(self,state):
  code,sample=state.h.unbind(-1);goal=(code>=2)|((code==1)&(sample<7));objects=torch.full((len(code),3,8),-20.);objects[:,:,0]=20.;objects[:,0,0]=torch.where(goal,-20.,20.);objects[:,0,3]=torch.where(goal,20.,-20.)
  drawer=torch.tensor([20.,-20.,-20.]).repeat(len(code),1);slider=drawer.clone();lights=torch.full((len(code),2),-20.)
  return {'objects':objects,'drawer':drawer,'slider':slider,'lights':lights}
 def outcome_logits(self,before,action,after):
  logits=torch.full((len(after.h),8),-20.);logits[:,0]=20.;bad=(after.h[:,0]==3)&(after.h[:,1]==7);logits[bad,3]=20.;return logits
model=FixtureModel();catalog=[{'skill':name,'family':'device'} for name in ['turn_on_led','turn_off_led','turn_on_lightbulb']];task=json.loads((P/'configs/tasks/dev/task_specs.json').read_text())[0];dfa=DFA(task['dfa']);root=State(torch.stack([torch.zeros(8),torch.arange(8)],-1),torch.zeros(8,1,1));progress=dfa.advance(dfa.initial,{'pink_moved':False})
planner=LearnedSearch(model,catalog,{'device':120},Calibration(threshold=.9,event_threshold=.9));result=planner.plan(root,task,dfa,progress)
# Independent Dijkstra over the fixture's explicitly known joint outcomes.
queue=[(0,())];expected=None
while queue:
 cost,sequence=heapq.heappop(queue)
 if sequence:
  last=sequence[-1]
  if last==2:continue
  if last==1:expected=sequence;break
 if cost==2:continue
 for action in range(3):heapq.heappush(queue,(cost+1,sequence+(action,)))
assert expected==(1,)
assert result['plan']==[catalog[i]['skill'] for i in expected],result['plan']
assert result['queries']==3 and result['sample_successor_evaluations']==24
assert result['edges'][0]['sample_goal_scores'][7]<.9 and all(x>.9 for x in result['edges'][0]['sample_goal_scores'][:7])
assert result['edges'][2]['rejection']=='process_constraint_violation'
assert result['edges'][2]['sample_event_probabilities'][7][2]>.9
record={'passed':True,'short_chain_Dijkstra_plan':list(expected),'beam_plan':result['plan'],'one_failed_sample_prevents_goal_acceptance':True,'one_constraint_violating_sample_rejects_entire_action_sequence':True,'logical_queries':3,'paired_sample_evaluations':24,'scope':'Explicit synthetic finite-graph unit fixture only; no real transition, trained-model accuracy, or physical success claimed'}
(P/'runs/core1200_v1/search_protocol_regression.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
