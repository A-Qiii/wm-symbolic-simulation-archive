from pathlib import Path
import json,collections,hashlib,csv
import numpy as np
P=Path('/home/robot/research/projects/wm_symbolic');A=P/'experiments/prediction_error_analysis_20261021';S=P/'experiments/data_expansion_training_20261021'
def read(p):return json.loads(p.read_text())
def save(n,rr):
 (A/f'results/{n}.json').write_text(json.dumps(rr,ensure_ascii=False,indent=2)+'\n')
 if rr:
  with(A/f'results/{n}.csv').open('w',newline='')as f:
   w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rr for k in r)));w.writeheader();w.writerows({k:json.dumps(v,ensure_ascii=False)if isinstance(v,(dict,list))else v for k,v in r.items()}for r in rr)
chains=read(A/'results/six_step_chains.json');logs={(r['source'],r['episode_id'],r['step']):r for r in read(A/'results/execution_log_audit.json')};rows=[];paired=[];diffs=[]
for source in ['old_development','development','combination_test']:
 for seed in [0,1]:
  models={}
  for arm in ['old228','new868']:
   z=dict(np.load(S/f'results/predictions_{source}_{arm}_seed{seed}.npz'));ix={(str(e),int(o),str(m),int(t)):i for i,(e,o,m,t)in enumerate(zip(z['episode_id'],z['origin'],z['mode'],z['relative_step']))};models[arm]=(z,ix)
   for r in chains:
    if(r['source'],r['arm'],r['seed'])!=(source,arm,seed)or r['first_error_step']<=0:continue
    eid=r['episode_id'];t=r['first_error_step'];i=ix[eid,0,'free',t];pr=ix[eid,0,'free',t-1];on=ix[eid,0,'one_step',t];log=logs[source,eid,t]
    q={**r,'GT_previous':z['cat'][pr,:4].tolist(),'GT_after':z['cat'][i,:4].tolist(),'prediction_previous_objects':z['prob'][pr,:24].reshape(3,8).argmax(1).tolist(),'prediction_after_objects':z['prob'][i,:24].reshape(3,8).argmax(1).tolist(),'prediction_before_drawer':int(z['prob'][pr,24:27].argmax()),'prediction_after_drawer':int(z['prob'][i,24:27].argmax()),'one_step_objects':z['prob'][on,:24].reshape(3,8).argmax(1).tolist(),'one_step_drawer':int(z['prob'][on,24:27].argmax()),'actual_success_label':log['actual_success_label'],'actual_controls':log['controls'],'actual_gripper_overrides':log['actual_gripper_overrides'],'target':int(z['target'][i])}
    q['target_was_In_and_stays_In']=bool(q['target']>=0 and q['GT_previous'][q['target']]==3 and q['GT_after'][q['target']]==3)
    q['target_was_In_predicted_Held']=bool(q['target_was_In_and_stays_In']and z['prob'][i,[6,14,22][q['target']]]>=.5)
    q['target_first_corrected_with_observation']=bool('target'in r['components']and r['one_step_observed_prefix_correct_at_free_first_error'])
    rows.append(q)
  old={r['episode_id']:r for r in chains if(r['source'],r['arm'],r['seed'])==(source,'old228',seed)};new={r['episode_id']:r for r in chains if(r['source'],r['arm'],r['seed'])==(source,'new868',seed)}
  for eid,a in old.items():
   b=new[eid]
   for metric in ['final_correct','whole_correct']:
    outcome='both_correct'if a[metric]and b[metric]else'gain'if b[metric]else'loss'if a[metric]else'both_wrong'
    paired.append({'source':source,'seed':seed,'episode_id':eid,'metric':metric,'outcome':outcome,'no_declared_intervention':b['no_declared_intervention'],'old_first_step':a['first_error_step'],'new_first_step':b['first_error_step'],'old_first_skill':a['first_error_skill'],'new_first_skill':b['first_error_skill'],'old_components':a['components'],'new_components':b['components'],'old_one_step_correct_at_first':a['one_step_observed_prefix_correct_at_free_first_error'],'new_one_step_correct_at_first':b['one_step_observed_prefix_correct_at_free_first_error']})
   # First boundary where old is joint right but new joint wrong; not necessarily new's first chain error.
   za,ia=models['old228'];zb,ib=models['new868']
   for t in range(7):
    u=ia[eid,0,'free',t];v=ib[eid,0,'free',t];c=za['cat'][u,:3];truth=np.r_[c==6,c==3];pa=np.r_[za['prob'][u,[6,14,22]],za['prob'][u,[3,11,19]]]>=.5;pb=np.r_[zb['prob'][v,[6,14,22]],zb['prob'][v,[3,11,19]]]>=.5;oka=np.array_equal(pa,truth)and int(za['prob'][u,24:27].argmax())==int(za['cat'][u,3]);okb=np.array_equal(pb,truth)and int(zb['prob'][v,24:27].argmax())==int(zb['cat'][v,3])
    if oka and not okb:
     ob=ib[eid,0,'one_step',t]if t else ib[eid,0,'current',0];po=np.r_[zb['prob'][ob,[6,14,22]],zb['prob'][ob,[3,11,19]]]>=.5;oneok=np.array_equal(po,truth)and int(zb['prob'][ob,24:27].argmax())==int(zb['cat'][ob,3]);diffs.append({'source':source,'seed':seed,'episode_id':eid,'first_old_right_new_wrong_step':t,'skill':str(zb['skill'][v]),'no_declared_intervention':b['no_declared_intervention'],'new_actual_prefix_one_step_correct':bool(oneok),'new_prior_step_was_chain_first_error':b['first_error_step'],'new_final_correct':b['final_correct']});break
save('first_errors_detailed',rows);save('six_step_paired_chains',paired);save('first_regression_boundary',diffs)
summary=[]
for so in ['old_development','development','combination_test']:
 for arm in ['old228','new868']:
  for se in [0,1]:
   rr=[r for r in rows if(r['source'],r['arm'],r['seed'])==(so,arm,se)and r['no_declared_intervention']];summary.append({'source':so,'arm':arm,'seed':se,'no_intervention_first_transition_errors':len(rr),'target_In_stays_In_error':sum(r['target_was_In_and_stays_In']and'target'in r['components']for r in rr),'target_In_predicted_Held':sum(r['target_was_In_predicted_Held']for r in rr),'target_first_fixed_by_observations':sum(r['target_first_corrected_with_observation']for r in rr),'drawer_first':sum('drawer'in r['components']for r in rr),'drawer_stable_but_pred_changed':sum('drawer'in r['components']and r['GT_previous'][3]==r['GT_after'][3]and r['prediction_before_drawer']!=r['prediction_after_drawer']for r in rr),'drawer_changed_actual_to_middle_pred_other':sum('drawer'in r['components']and r['GT_previous'][3]!=r['GT_after'][3]and r['GT_after'][3]==1 for r in rr)})
save('mechanism_summary',summary)
print(json.dumps(summary,ensure_ascii=False))
