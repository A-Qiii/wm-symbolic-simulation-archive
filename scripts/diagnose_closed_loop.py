"""Single-interface counterfactuals, explicitly isolated from online planning."""
from pathlib import Path
import json,gzip
import torch
from wm_symbolic.world_model import WorldModel,State
from wm_symbolic.search import DFA,LearnedSearch
from wm_symbolic.symbol_decoder import Calibration,probabilities,goal_score
from wm_symbolic.dataset import static_arguments
from wm_symbolic.gpu_phase import gpu_phase
P=Path('/home/robot/research/projects/wm_symbolic');run=P/'runs/closed_loop_interface_C000';runtime=json.loads((run/'runtime.json').read_text());scoring=json.loads((run/'independent_score.json').read_text());task=runtime['task']
ck=torch.load(runtime['checkpoint'],map_location='cpu',weights_only=False);model=WorldModel(event_count=7).eval();model.load_state_dict(ck['model']);torch.set_num_threads(4)
catalog=json.loads((P/'configs/skills.json').read_text());budgets=json.loads((P/'configs/executor_budgets_pilot.json').read_text())['steps'];dfa=DFA(task['dfa']);cal=Calibration();planner=LearnedSearch(model,catalog,budgets,cal)
root=torch.load(run/'planning_root_00.pt',map_location='cpu',weights_only=False);original=json.load(gzip.open(run/'plan_00.json.gz','rt'))
gt=scoring['boundary_trace'][0]['facts'];progress=dfa.advance(dfa.initial,{n:gt[n] for n in dfa.props});root_goal=all(gt[n]==wanted for n,wanted in task['terminal_goals'].items())
with gpu_phase(P,'single_interface_diagnostics'),torch.inference_mode():
 model.cuda();torch.set_rng_state(root['cpu_rng']);torch.cuda.set_rng_state(root['cuda_rng'].cpu())
 changed=planner.plan(State(root['h'].cuda(),root['z'].cuda()),task,dfa,progress)
 late=torch.load(run/'planning_root_02.pt',map_location='cpu',weights_only=False);state=State(late['h'].cuda(),late['z'].cuda());idx=next(i for i,x in enumerate(catalog) if x['skill']=='place_in_drawer')
 torch.set_rng_state(late['cpu_rng']);torch.cuda.set_rng_state(late['cuda_rng'].cpu())
 with torch.autocast('cuda',dtype=torch.bfloat16):
  action=model.action(torch.full((8,),idx,device='cuda',dtype=torch.long),torch.tensor([static_arguments('place_in_drawer')]*8,device='cuda'),torch.full((8,),budgets[catalog[idx]['family']],device='cuda',dtype=torch.float32))
  after,_=model.imagine(state,action);p=probabilities(model.symbols(after),cal);e=model.outcome_logits(state,action,after).float().sigmoid().mean(0)
 diagnostic={'prior_red_drawer_probability':float(p['objects'][:,0,3].mean()),'prior_pink_moved_probability':float(e[3]),
             'actual_red_drawer':scoring['boundary_trace'][-1]['facts']['red_drawer'],'actual_pink_moved':scoring['boundary_trace'][-1]['facts']['pink_moved']}
with gzip.open(run/'root_label_replacement_search.json.gz','wt') as f:json.dump(changed,f)
result={'root_label_replacement':{'same_latent_h_z':True,'same_model_actions_budget_and_rng':True,'root_goal_true':root_goal,
        'original_plan':original['plan'],'replacement_plan':changed['plan'],'plan_unchanged':original['plan']==changed['plan'],
        'interpretation':'Only root labels changed. Unchanged plan does not repair or validate latent state; no task recovery claimed.'},
        'actual_history_one_step_restart_before_place':diagnostic,
        'first_decision_relevant_error':'The selected plan predicts placing red into the closed drawer from a real history where red has not been held; the last one-step restart still predicts the false outcome and misses pink motion.',
        'no_handcoded_dynamic_preconditions_added':True,'offline_diagnostic_only':True}
(run/'single_interface_diagnostics.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
