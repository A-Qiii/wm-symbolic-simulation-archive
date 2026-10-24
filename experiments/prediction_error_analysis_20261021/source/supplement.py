from pathlib import Path
import json,collections,csv
import numpy as np
P=Path('/home/robot/research/projects/wm_symbolic');A=P/'experiments/prediction_error_analysis_20261021';S=P/'experiments/data_expansion_training_20261021';O=P/'experiments/object_recurrent_state_run_20261019'
def rd(n):return json.loads((A/f'results/{n}.json').read_text())
def out(n,rr):
 (A/f'results/{n}.json').write_text(json.dumps(rr,ensure_ascii=False,indent=2)+'\n')
 if rr:
  with(A/f'results/{n}.csv').open('w',newline='')as f:
   w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rr for k in r)));w.writeheader();w.writerows(rr)
chs=rd('six_step_chains');orig=rd('origin_endpoints');logs=rd('execution_log_audit');pairs=rd('external_ambiguity_pairs');diagn=[];strict=[];one=[];ceil=[]
for key in sorted({(r['source'],r['arm'],r['seed'])for r in chs}):
 for cohort in ['all','no_declared_intervention']:
  rr=[r for r in chs if(r['source'],r['arm'],r['seed'])==key and r['first_error_step']>0 and(cohort=='all'or r['no_declared_intervention'])]
  cc=collections.Counter((r['one_step_observed_prefix_correct_at_free_first_error'],r['current_posterior_correct_at_free_first_error'])for r in rr)
  diagn.append({'source':key[0],'arm':key[1],'seed':key[2],'cohort':cohort,'free_first_transition_errors':len(rr),'one_correct_current_correct':cc[True,True],'one_correct_current_wrong':cc[True,False],'one_wrong_current_correct':cc[False,True],'one_wrong_current_wrong':cc[False,False]})
for key in sorted({(r['arm'],r['seed'])for r in orig}):
 rr=[r for r in orig if r['source']=='combination_test'and(r['arm'],r['seed'])==key and r['horizon']==3 and r['target_combo']]
 strict.append({'source':'combination_test','arm':key[0],'seed':key[1],'origins':len(rr),'endpoint_eligible':sum(r['endpoint_eligible']for r in rr),'endpoint_correct':sum(r['endpoint_eligible']and r['endpoint_correct']for r in rr),'future_interval_eligible':sum(r['future_all_valid']for r in rr),'future_interval_correct':sum(r['future_all_valid']and r['future_all_correct']for r in rr),'including_initial_interval_eligible':sum(r['including_initial_all_valid']for r in rr),'including_initial_interval_correct':sum(r['including_initial_all_valid']and r['including_initial_all_correct']for r in rr)})
for source in ['old_development','development','combination_test']:
 epath=O/'cache/development_metadata.json'if source=='old_development'else S/f'cache/{source}_metadata.json';es=json.loads(epath.read_text());kinds={e['episode_id']:e['kind']for e in es};faults=collections.defaultdict(set)
 for r in logs:
  if r['source']==source and r['declared_controls']:faults[r['episode_id']].add(r['step'])
 for arm in ['old228','new868']:
  for seed in [0,1]:
   z=dict(np.load(S/f'results/predictions_{source}_{arm}_seed{seed}.npz'));p=z['prob'];pred=np.c_[p[:,[6,14,22]]>=.5,p[:,[3,11,19]]>=.5];truth=np.c_[z['cat'][:,:3]==6,z['cat'][:,:3]==3];eligible=z['cat_valid'][:,:4].all(1);ok=(pred==truth).all(1)&(p[:,24:27].argmax(1)==z['cat'][:,3])
   for kind in sorted(set(kinds.values())):
    for selectname in ['one_step_all','one_step_excluding_fault_boundary','one_step_final6','current_all']:
     sel=(z['origin']==0)&eligible&np.array([kinds[str(e)]==kind for e in z['episode_id']])
     sel &=z['mode']==('current'if selectname=='current_all'else'one_step')
     if selectname=='one_step_excluding_fault_boundary':sel &=np.array([int(t)not in faults[str(e)]for e,t in zip(z['episode_id'],z['relative_step'])])
     if selectname=='one_step_final6':sel &=(z['relative_step']==6)&(z['length']==6)
     one.append({'source':source,'arm':arm,'seed':seed,'kind':kind,'scope':selectname,'eligible':int(sel.sum()),'correct':int((sel&ok).sum())})
   rr=[r for r in pairs if(r['source'],r['arm'],r['seed'])==(source,arm,seed)];n=sum(e['length']==6 for e in es);verified=sum(r['ambiguous_verified']for r in rr);ceil.append({'source':source,'arm':arm,'seed':seed,'six_step_episodes':n,'declared_intervention_pairs':len(rr),'identical_input_opposite_GT_pairs':verified,'maximum_whole_chain_correct_due_to_these_pairs':n-verified,'actual_input_prefix_same_pairs':sum(r['public_prefix_features_exact']for r in rr),'exact_prediction_pairs':sum(r['predictions_exact']for r in rr),'bound_not_for_final_endpoint':True})
out('observation_prefix_error_partition',diagn);out('strict_combination_three_step',strict);out('observed_comparison',one);out('external_information_bound',ceil)
# Internal counting checks and exact correspondence to original endpoint metrics.
metrics=json.loads((S/'results/metrics.json').read_text());count=rd('three_step_counting')
for r in count:
 q=next(x for x in metrics if(x['source'],x['arm'],x['seed'],x['scope'])==(r['source'],r['arm'],r['seed'],'free_h3'));assert q['joint_correct']==r['endpoint_correct']and q['joint_eligible']==r['endpoint_eligible']
 assert r['including_initial_interval_correct']<=r['future_interval_correct']<=r['endpoint_correct']
for r in strict:
 q=next(x for x in metrics if(x['source'],x['arm'],x['seed'],x['scope'])==(r['source'],r['arm'],r['seed'],'heldout_combo_free_h3'));assert q['joint_correct']==r['endpoint_correct']and q['joint_eligible']==r['endpoint_eligible']
for r in rd('six_step_first_error_summary'):
 assert sum(r[k]for k in ['initial_first','target_only_first','non_target_stable_only_first','drawer_only_first','mixed_first','non_target_GT_change_only_first','no_error'])==r['episodes']
for r in diagn:assert sum(r[k]for k in ['one_correct_current_correct','one_correct_current_wrong','one_wrong_current_correct','one_wrong_current_wrong'])==r['free_first_transition_errors']
(A/'results/counting_verification.json').write_text(json.dumps({'passed':True,'three_step_original_rows_reconciled':len(count),'strict_original_rows_reconciled':len(strict),'mutually_exclusive_first_error_categories_sum_to_episodes':True,'observation_prefix_partitions_sum_to_transition_errors':True,'all_joint_labels_valid_in_all_six_step_chains':all(r['whole_all_valid']for r in chs),'six_step_chains':len(chs)},indent=2)+'\n')
print('STRICT',strict);print('OLD ROLE OBS',[r for r in one if r['source']=='old_development'and r['kind']=='role']);print('BOUNDS',ceil)
