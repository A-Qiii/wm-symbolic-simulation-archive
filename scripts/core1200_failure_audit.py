"""Locate observable failure boundaries without altering any experiment."""
from pathlib import Path
import json,torch
from wm_symbolic.symbol_decoder import Calibration,decode
from wm_symbolic.search import DFA
P=Path('/home/robot/research/projects/wm_symbolic');R=P/'runs/core1200_v1'
def audit_failures():
 tasks=json.loads((R/'evaluation/tasks.json').read_text());cal=Calibration(**json.loads((R/'evaluation/calibration/calibration.json').read_text())['calibration']);rows=[]
 for method in ['learned','goal_only','reference']:
  for task in tasks:
   d=R/'evaluation/closed_loop'/method/task['task_id'];result=json.loads((d/'result.json').read_text())
   if result['physical_task_success']:continue
   runtime=json.loads((d/'runtime.json').read_text());score=json.loads((d/'score_private.json').read_text());dfa=DFA(task['dfa']);entry={'method':method,'task_id':task['task_id'],'status':result['status'],'independent_score':{'terminal_goal':score['terminal_goal'],'LTLf_satisfied':score['LTLf_satisfied']},'evidence':[],'causal_uniqueness_claimed':False}
   actual={}
   for step in score['trace']:actual[step['boundary']]=step
   for step in score['trace']:
    if step['dfa_state'] not in dfa.distance:
     entry['evidence'].append({'boundary':step['boundary'],'kind':'actual_process_violation','facts':step['facts'],'request':step.get('request')});break
   relevant=sorted(set(task['dfa']['propositions'])|set(task['terminal_goals']))
   # Preserve both the original estimate and any reobservation revision.
   for step in runtime['trace']:
    if 'belief' not in step:continue
    boundary=step.get('boundary',0);belief=step['belief'];probs={k:torch.tensor(v).mean(0) for k,v in belief['symbol_probabilities'].items()};events=torch.tensor(belief['event_probabilities']).mean(0);estimated,reason=decode(probs,events,relevant,cal);truth=actual[boundary]['facts'];different={n:{'estimated':estimated[n],'actual':truth.get(n)} for n in relevant if truth.get(n) is not None and estimated[n]!=truth[n]}
    if different:
     entry['evidence'].append({'boundary':boundary,'kind':'current_readout_unknown' if reason else 'current_relevant_fact_error','reobserved':bool(step.get('reobserve')),'differences':different,'same_boundary_revision_may_recover_this_error':not bool(step.get('reobserve'))})
   if result['status']=='no_plan_within_budget':
    import gzip
    files=sorted(d.glob('plan_*.json.gz'))
    if files:
     with gzip.open(files[-1],'rt') as f:plan=json.load(f)
     entry['evidence'].append({'boundary':result['physical_skills'],'kind':'learned_successors_rejected_or_budget_exhausted','queries':plan['queries'],'rejections':plan['rejections'],'not_proof_of_physical_infeasibility':True})
   for step in score['trace'][1:]:
    if step.get('result',{}).get('status')=='failed':
     entry['evidence'].append({'boundary':step['boundary'],'kind':'controller_reported_failure','request':step['request'],'controller_reason':step['result'].get('reason'),'actual_facts':step['facts'],'status_alone_not_proof_of_task_failure':True});break
   if not entry['evidence']:entry['evidence'].append({'boundary':result['physical_skills'],'kind':'terminal_goal_not_reached_within_fixed_protocol','status':result['status']})
   entry['first_observed_difference']=min(x['boundary'] for x in entry['evidence']);entry['stopping_boundary']=result['physical_skills'];rows.append(entry)
 report={'failed_tasks':len(rows),'rows':rows,'scope':'Descriptive first differences and stopping evidence; a recovered transient readout error is not automatically causal. Paired single-module intervention is reported separately. No weights or thresholds changed.'}
 (R/'evaluation/failure_audit.json').write_text(json.dumps(report,indent=2));return report
