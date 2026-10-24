from pathlib import Path
import json,torch
from wm_symbolic.symbol_decoder import Calibration
from wm_symbolic.proposition_interface import decode,required_facts,action_observations
R=Path('/home/robot/research/projects/wm_symbolic/runs/proposition_interface_20261014')
c=Calibration(threshold=.8,event_threshold=.9);p={'objects':torch.tensor([[.33,.33,.31,.01,.005,.005,.005,.005]]*3),'drawer':torch.tensor([.03,.03,.94]),'slider':torch.tensor([.98,.01,.01]),'lights':torch.zeros(2)};e=torch.zeros(3)
names=['red_table','red_drawer','red_held','blue_held','pink_held','hand_empty'];old,oe=decode(p,e,names,c,'legacy');new,ne=decode(p,e,names,c,'proposition')
assert all(old[n] is None for n in names);assert new==dict(red_table=True,red_drawer=False,red_held=False,blue_held=False,pink_held=False,hand_empty=True) and ne is None
assert decode(p,e,['red_table_left'],c)[0]['red_table_left'] is None
p2={k:v.clone() for k,v in p.items()};p2['objects'][0]=torch.tensor([.1,0,0,0,0,0,.9,0]);p2['objects'][1]=p2['objects'][0]
assert decode(p2,e,['pink_drawer'],c)[1]=='multiple_held_conflict'
p3={k:v.clone() for k,v in p.items()};p3['objects'][0]=torch.tensor([.4,0,0,0,0,0,.6,0]);assert decode(p3,e,['hand_empty','red_held'],c)[0]=={'hand_empty':None,'red_held':None}
p4={k:v.clone() for k,v in p.items()};p4['objects'][0]=torch.tensor([.01,0,0,.99,0,0,0,0]);f,err=decode(p4,e,action_observations('lift_red_block_table'),c);assert f['red_table'] is False and err is None
scope=required_facts({'terminal_goals':{'red_drawer':True},'dfa':{'propositions':['pink_moved','slider_right']}});assert {'red_drawer','pink_moved','slider_right','red_held','blue_held','pink_held','hand_empty'}==set(scope)
out={'passed':True,'coarse_table_marginal':.97,'coarse_false_drawer_not_blocked_by_fine_uncertainty':True,'explicit_fine_goal_remains_unknown':True,'global_multiple_held_conflict_preserved_for_non_goal_objects':True,'ambiguous_held_does_not_settle_handempty':True,'false_action_precondition_is_known_not_pruned':True,'goals_constraints_global_held_scope_preserved':True,'scope':'Synthetic decoder/scope tests, not empirical planning success.'};(R/'evaluation/interface_regression.json').write_text(json.dumps(out,indent=2));print(out)
