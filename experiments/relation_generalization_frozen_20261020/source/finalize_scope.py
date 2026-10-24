from common import *
import fcntl
import finalize
guard();verify();lock=(P/'runs/gpu.lock').open();fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert read(S/'results/runtime_qa.json')['passed']
assert read(S/'results/scope_limitations.json')['executor_reads_private_GT']
assert read(S/'results/initial_relation_diagnostic.json')['original_cold_start_scores_not_replaced']
for name,source in [('example_display_frozen.json','example.py'),('held_display_frozen.json','held_discrimination.py'),('information_check_frozen.json','paired_information.py')]:
 assert read(S/'configs'/name)['source_sha256']==sha(S/'source'/source)
original=finalize.save
def save_scope(path,value):
 if Path(path).name in ['final_verification.json','SEALED.json']:
  value.update(executor_uses_existing_private_state=True,entire_system_GT_only_initialization_and_evaluation=False,world_model_private_GT_inputs=False,hold_scope='Observed completion controls4/8/12; no independent hold action or added controls',primary_scores_unchanged_by_posthoc_prior_diagnostic=True,initial_In_generalization_unresolved=True)
 return original(path,value)
finalize.save=save_scope
finalize.main()
