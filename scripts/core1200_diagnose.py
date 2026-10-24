"""One root-readout replacement with the exact saved latent/RNG; real reexecution."""
from pathlib import Path
import json
import numpy as np
from core1200_closed_loop import run,facts
P=Path('/home/robot/research/projects/wm_symbolic');R=P/'runs/core1200_v1';base=R/'evaluation/closed_loop'
def main():
 from core1200_failure_audit import audit_failures
 audit_failures()
 tasks=json.loads((R/'evaluation/tasks.json').read_text());candidates=[]
 for task in tasks:
  path=base/'learned'/task['task_id'];ref=base/'reference'/task['task_id']
  if not (path/'result.json').exists() or not (ref/'result.json').exists():continue
  result=json.loads((path/'result.json').read_text());reference=json.loads((ref/'result.json').read_text())
  if not result['physical_task_success'] and reference['physical_task_success']:candidates.append((task,path,result,reference))
 if not candidates:
  (R/'evaluation/diagnosis.json').write_text(json.dumps({'performed':False,'reason':'No learned failure with successful same-root reference; no unsupported correction performed.'},indent=2));return
 task,path,result,reference=candidates[0];runtime=json.loads((path/'runtime.json').read_text());root=json.loads((path/'execution/evaluator_only/root.json').read_text());truth=facts(root['label'],[0,0,0]);belief=runtime['trace'][0]['belief']['symbol_probabilities']
 predicted=[int(np.array(belief['objects']).mean(0)[j].argmax()) for j in range(3)]+[int(np.array(belief[k]).mean(0).argmax()) for k in ['drawer','slider']]
 confidence=[float(np.array(belief['objects']).mean(0)[j].max()) for j in range(3)]+[float(np.array(belief[k]).mean(0).max()) for k in ['drawer','slider']]
 relevant=[n for n in set(task['dfa']['propositions'])|set(task['terminal_goals']) if truth.get(n) is not None]
 estimated=next((x['facts'] for x in runtime['trace'] if 'facts' in x),None)
 source=path/'planning_root_00_r0.pt'
 if not source.exists():raise FileNotFoundError('Exact pre-error latent is required for paired diagnosis')
 diag=run(task,'learned',[],root_override=truth,root_latent=source)
 report={'performed':True,'task_id':task['task_id'],'first_examined_boundary':0,'original_status':result['status'],'actual_root_categories':root['label']['categorical'],'learned_root_argmax_categories':predicted,'learned_root_max_probabilities':confidence,'relevant_truth':{n:truth[n] for n in relevant},'estimated_facts_if_decoded':estimated,'replacement':'Only root symbolic labels; identical saved h/z and RNG; successor model, controller, budgets, threshold, and later observations unchanged.','original_physical_success':result['physical_task_success'],'reference_physical_success':reference['physical_task_success'],'replacement_result':diag,'recovered_plan':diag['physical_skills']>result['physical_skills'] or diag['physical_task_success'],'recovered_task':diag['physical_task_success'],'model_weights_changed':False,'test_used_for_training_or_threshold_selection':False,'interpretation':'Plan recovery is counted separately from actual task recovery. A failed replacement does not rule out errors already encoded in the latent state. This diagnostic is excluded from primary test success counts.'}
 (R/'evaluation/diagnosis.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
