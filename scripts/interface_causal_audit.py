from pathlib import Path
from collections import Counter
import json,gzip
import numpy as np
P=Path('/home/robot/research/projects/wm_symbolic');R=P/'runs/proposition_interface_20261014'
def read(p):return json.loads(p.read_text())
refs={x['task_id']:x['reference'] for x in read(R/'references_evaluator_only.json')};result=read(R/'evaluation/RESULT.json');rows=[]
for res in result['results']:
 if res['cohort']!='fresh_D' or res['method']!='proposition' or res['status']!='no_plan_within_budget':continue
 dest=R/'evaluation/closed_loop/proposition'/res['task_id'];attempts=[]
 for file in sorted(dest.glob('plan_*.json.gz')):
  d=json.load(gzip.open(file,'rt'));edges=d['edges'];depth=max(e['depth'] for e in edges);survivors=[e for e in edges if e['depth']==depth and e['rejection'] is None];termination='query_limit' if d['queries']>=5000 else 'horizon_limit_with_surviving_candidates' if depth==8 and survivors else 'no_surviving_frontier'
  unknown=Counter();stages=Counter()
  for e in edges:
   detail=e.get('failed_details')
   if detail:
    stages[detail['stage']]+=1
    for name,value in detail['facts'].items():
     if value is None:unknown[name]+=1
  path=[];node=0
  for skill in refs[res['task_id']]:
   e=next((x for x in edges if x['parent_id']==node and x['skill']==skill),None)
   if e is None:break
   path.append({'skill':skill,'depth':e['depth'],'rejection':e['rejection'],'failed_details':e.get('failed_details'),'sample_goal_scores':e['sample_goal_scores']})
   if e['rejection']:break
   node=e['node_id']
  attempts.append({'file':str(file),'queries':d['queries'],'query_limit':5000,'max_explored_depth':depth,'horizon_limit':8,'surviving_edges_at_last_depth':len(survivors),'termination_inferred_from_logged_edges_and_frozen_search':termination,'rejections':d['rejections'],'rejecting_stage_counts':dict(stages),'unknown_fact_occurrences_among_first_failed_samples':dict(unknown),'reference_prefix_prediction':path,'false_precondition_value_rejections':0,'mutual_exclusion_rejections':sum(v for k,v in d['rejections'].items() if 'multiple_held_conflict' in k),'interpretation':'False values of action-related observations are never used to prune. UNKNOWN is insufficient evidence under frozen thresholds, not proof that the predicted argmax or physical transition is wrong. Query/horizon bounds are stated separately.'})
 rows.append({'task':res['task_id'],'actual_skill_requests':res['physical_skills'],'root_current_readout_wrong':next(x for x in result['failure_evidence'] if x['task']==res['task_id'] and x['method']=='proposition')['first_current_wrong'],'attempts':attempts})
example_dir=R/'evaluation/closed_loop/proposition/C_core1200_01_red_closed';runtime=read(example_dir/'runtime.json');belief=next(x['belief'] for x in runtime['trace'] if 'belief' in x);objects=np.array(belief['symbol_probabilities']['objects']).mean(0)
example={'task':'C_core1200_01_red_closed','physical_success':True,'skills':3,'root_margins':{c:{'fine_class_max':float(objects[j].max()),'table_sum':float(objects[j,:3].sum()),'drawer':float(objects[j,3]),'held':float(objects[j,6])} for j,c in enumerate(['red','blue','pink'])},'explanation':'Identical state and probabilities: old fine-class gate blocks required object facts; new proposition margins permit their coarse true/false readings. Does not independently isolate adding table aggregation, because both matched modes expose the same table marginal.'}
out={'scope':'Read-only post-run causal-stage audit; no reexecution or new threshold choices.','matched_factor':'Both modes use same proposition marginals (including table sum), same task/action-observability scope, temperatures,0.95/0.9 thresholds, global held checks, eight-sample search and frozen model. Only the legacy max-category gate differs.','historical_baseline_confound':'Compared with historical core1200, the common fact scope adds all held/HandEmpty and candidate pre-action observability, tasks add slider_right preservation, and symbol threshold changes0.8->0.95. These common changes are NOT separately ablated; historical1/12 and current0/8 cannot estimate the decoder effect.','identifiable_contribution':'Within this matched complete interface, removing the fine-category gate rescues25 correct initial facts onABC and27 onnewD with0 incorrect rescues. Only oneABC task physically recovers; noD task recovers.','aggregation_contribution_not_separately_identified':True,'no_plan_cases':rows,'representative_success':example,'newD_used_to_retune':False}
(R/'evaluation/causal_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
print(json.dumps({'example':example,'cases':[{'task':x['task'],'attempts':[{k:a[k] for k in ['queries','max_explored_depth','termination_inferred_from_logged_edges_and_frozen_search','rejections','rejecting_stage_counts','mutual_exclusion_rejections']} for a in x['attempts']]} for x in rows]},ensure_ascii=False,indent=2))
