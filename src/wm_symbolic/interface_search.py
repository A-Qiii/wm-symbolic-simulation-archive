"""Finite TS edges come exclusively from learned RSSM successor calls."""
from dataclasses import dataclass
from collections import deque,Counter
import torch
from wm_symbolic.world_model import State
from wm_symbolic.symbol_decoder import probabilities,event_probabilities
from wm_symbolic.proposition_interface import decode,goal_score,required_facts,action_observations
from wm_symbolic.dataset import static_arguments
from wm_symbolic.legacy_dfa import reuse_compiled_automaton

class DFA:
 def __init__(self,data):
  self.props=tuple(data['propositions']);self.initial=int(data['initial']);self.accepting=set(data['accepting']);self.transitions=data['transitions']
  self.monitor=reuse_compiled_automaton(data)
  self.distance={s:0 for s in self.accepting};queue=deque(self.accepting)
  while queue:
   node=queue.popleft()
   for src,guard,dst in self.transitions:
    if dst==node and src not in self.distance:self.distance[src]=self.distance[node]+1;queue.append(src)
 def advance(self,state,facts):
  if any(type(facts.get(n)) is not bool for n in self.props):raise ValueError('UNKNOWN DFA proposition')
  return self.monitor.advance((state,False,False,False,()),dict(facts,done=False))[0]

@dataclass
class Node:
 state:State
 progress:tuple
 sequence:tuple
 score:tuple
 path_id:int

class InterfaceSearch:
 def __init__(self,model,catalog,budgets,calibration,beam=32,horizon=8,query_limit=5000,samples=8,mode="proposition"):
  self.mode=mode;self.model=model;self.catalog=catalog;self.budgets=budgets;self.calibration=calibration;self.beam=beam;self.horizon=horizon;self.query_limit=query_limit;self.samples=samples
  if (beam,horizon,query_limit,samples)!=(32,8,5000,8):raise ValueError('Main search protocol constants differ')
 @torch.no_grad()
 def plan(self,root,task,dfa,actual_progress,ignore_ltl=False):
  assert root.h.shape[0]==self.samples
  self.model.eval();device=root.h.device;n=len(self.catalog);queries=0;next_id=1;edges=[];rejections=Counter()
  frontier=[Node(root,tuple([actual_progress]*self.samples),(),(0.,0.,0),0)]
  skills=torch.arange(n,device=device);args=torch.tensor([static_arguments(s['skill']) for s in self.catalog],device=device);budgets=torch.tensor([self.budgets[s['family']] for s in self.catalog],device=device,dtype=torch.float32)
  with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):all_actions=self.model.action(skills,args,budgets)
  for depth in range(1,self.horizon+1):
   candidates=[];accepted=[]
   for parent in frontier:
    count=min(n,self.query_limit-queries)
    if count<=0:break
    state=State(parent.state.h.repeat(count,1),parent.state.z.repeat(count,1,1));actions=all_actions[:count].repeat_interleave(self.samples,0)
    with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
     parent_probs=probabilities(self.model.symbols(parent.state),self.calibration)
     child,_=self.model.imagine(state,actions);pred=self.model.symbols(child);events=event_probabilities(self.model.outcome_logits(state,actions,child).float()[:,1:],self.calibration)
    probs=probabilities(pred,self.calibration);scores=goal_score(probs,task['terminal_goals'],self.calibration).reshape(count,self.samples)
    for action_id in range(count):
     queries+=1;offset=action_id*self.samples;progress=[];reason=None;failed_details=None
     for sample in range(self.samples):
      pre,pre_error=decode({k:v[sample] for k,v in parent_probs.items()},torch.zeros(3,device=device),action_observations(self.catalog[action_id]['skill']),self.calibration,self.mode)
      if pre_error:
       reason='candidate_input_'+pre_error;failed_details={'stage':'parent_action_observability','sample':sample,'facts':pre,'objects':parent_probs['objects'][sample].tolist()};break
      sample_probs={k:v[offset+sample] for k,v in probs.items()};facts,conflict=decode(sample_probs,events[offset+sample],required_facts(task),self.calibration,self.mode)
      if conflict=='multiple_held_conflict' or (conflict and not ignore_ltl):
       reason=conflict;failed_details={'stage':'imagined_successor','sample':sample,'facts':facts,'objects':sample_probs['objects'].tolist(),'drawer':sample_probs['drawer'].tolist(),'events':events[offset+sample].tolist()};break
      if ignore_ltl:progress.append(parent.progress[sample]);continue
      successor=dfa.advance(parent.progress[sample],facts)
      if successor not in dfa.distance:reason='process_constraint_violation';break
      progress.append(successor)
     node_id=next_id;next_id+=1;seq=parent.sequence+(self.catalog[action_id]['skill'],)
     edge={'node_id':node_id,'parent_id':parent.path_id,'skill':self.catalog[action_id]['skill'],'depth':depth,'failed_details':failed_details,'sample_goal_scores':scores[action_id].tolist(),'sample_dfa_states':progress,'rejection':reason,
       'sample_object_locations':probs['objects'][offset:offset+self.samples].argmax(-1).tolist(),
       'sample_drawer_states':probs['drawer'][offset:offset+self.samples].argmax(-1).tolist(),
       'sample_slider_states':probs['slider'][offset:offset+self.samples].argmax(-1).tolist(),
       'sample_light_probabilities':probs['lights'][offset:offset+self.samples].tolist(),
       'sample_event_probabilities':events[offset:offset+self.samples].tolist(),
       'interface_mode':self.mode,'required_successor_facts':required_facts(task),'precondition_values_not_used_as_action_rules':True,'transition_source':getattr(self.model,'transition_source','RSSM.imagine'),'latent_states_merged':False}
     edges.append(edge)
     if reason:rejections[reason]+=1;continue
     # Keep all eight paired outcomes together; the planner cannot choose one.
     score=(-sum(dfa.distance.get(s,0) for s in progress)/self.samples,float(scores[action_id].mean()),-depth)
     node=Node(State(child.h[offset:offset+self.samples],child.z[offset:offset+self.samples]),tuple(progress),seq,score,node_id)
     candidates.append(node)
     goal_confirmed=bool((scores[action_id]>=self.calibration.threshold).all())
     if goal_confirmed and (ignore_ltl or all(s in dfa.accepting for s in progress)):accepted.append(node)
   if accepted:
    best=max(accepted,key=lambda x:x.score);return {'status':'candidate','plan':list(best.sequence),'queries':queries,'sample_successor_evaluations':queries*self.samples,'edges':edges,'rejections':dict(rejections),
      'ranking_score':best.score,'calibrated':self.calibration.calibrated,'not_a_safety_guarantee':True}
   frontier=sorted(candidates,key=lambda x:x.score,reverse=True)[:self.beam]
   if not frontier or queries>=self.query_limit:break
  return {'status':'unsolved_within_budget','plan':[],'queries':queries,'sample_successor_evaluations':queries*self.samples,'edges':edges,'rejections':dict(rejections),'not_proof_of_infeasibility':True,'calibrated':self.calibration.calibrated}
