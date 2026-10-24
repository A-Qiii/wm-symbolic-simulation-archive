"""Offline family NLL, changed facts and paired-context accuracy, no model tuning."""
from pathlib import Path
import argparse,json,math,collections,hashlib,csv
import numpy as np
a=argparse.ArgumentParser();a.add_argument('--input',type=Path,required=True);args=a.parse_args();rows=[json.loads(x) for x in (args.input/'predictions.jsonl').read_text().splitlines()];analysis=json.loads((args.input/'analysis.json').read_text())
families=['red_location','blue_location','pink_location','drawer','slider','light','led'];classes=[8,8,8,3,3,2,2]
def probability(record,j):
 p=record['symbol_probabilities']
 if j<3:return np.asarray(p['objects'][j],float)
 if j<5:return np.asarray(p['drawer' if j==3 else 'slider'],float)
 v=float(p['lights'][j-5]);return np.asarray([1-v,v])
conditions={'current_observation':('current',None),'real_history_one_step':('one_step_real_history',None),'root_free_1':('free_from_root',0),'root_free_3':('free_from_root',2),'root_free_5':('free_from_root',4)};table=[];joint=[]
for condition,(kind,step) in conditions.items():
 selected=[r for r in rows if step is None or r['step']==step];joint_valid=[r for r in selected if all(r[kind]['valid'])]
 joint.append({'condition':condition,'boundaries':len(selected),'fully_scored_boundaries':len(joint_valid),'whole_seven_fact_accuracy':sum(all(r[kind]['correct']) for r in joint_valid)/len(joint_valid) if joint_valid else None,'scoring_masked_boundaries':len(selected)-len(joint_valid)})
 for j,name in enumerate(families):
  conf=np.zeros((classes[j],classes[j]),int);nll=[];changed=[];unchanged=[];unknown=0
  for r in selected:
   item=r[kind]
   if not item['valid'][j]:continue
   truth=item['target'][j];guess=item['predicted'][j];conf[truth,guess]+=1;nll.append(-math.log(max(float(probability(item,j)[truth]),1e-9)));unknown+=item['confidence'][j]<.5
   if kind!='current':(changed if r['changed'][j] else unchanged).append(truth==guess)
  tp=np.diag(conf);den=conf.sum(0)+conf.sum(1);f1=np.divide(2*tp,den,out=np.zeros_like(tp,dtype=float),where=den>0)
  table.append({'condition':condition,'predicate_family':name,'valid_facts':int(conf.sum()),'NLL':float(np.mean(nll)) if nll else None,'accuracy':float(tp.sum()/conf.sum()) if conf.sum() else None,'macro_F1_present_classes':float(f1[den>0].mean()) if (den>0).any() else None,'unknown_rate_at_diagnostic_0_5':unknown/len(nll) if nll else None,'changed_facts':len(changed),'changed_accuracy':float(np.mean(changed)) if changed else None,'unchanged_facts':len(unchanged),'unchanged_accuracy':float(np.mean(unchanged)) if unchanged else None})
context=collections.defaultdict(list)
for r in rows:context[(r['root_id'],r['skill'])].append(r)
hashes={}
def public_hash(path):
 if path not in hashes:
  with np.load(path) as z:hashes[path]=hashlib.sha256(b''.join(z[k].tobytes() for k in ['rgb_static','rgb_gripper','robot_obs'])).hexdigest()
 return hashes[path]
pairs=[]
for (root,skill),items in context.items():
 branches={}
 for r in sorted(items,key=lambda x:x['step']):branches.setdefault(r['episode_id'],r)
 unique=list(branches.values())
 for i,left in enumerate(unique):
  for right in unique[i+1:]:
   if public_hash(left['before_public'])==public_hash(right['before_public']):continue
   li=left['one_step_real_history'];ri=right['one_step_real_history'];valid=all(li['valid']) and all(ri['valid'])
   symbolic_difference=any(x!=y and a and b for x,y,a,b in zip(left['current']['target'],right['current']['target'],left['current']['valid'],right['current']['valid']))
   pairs.append({'root_id':root,'skill':skill,'left_episode':left['episode_id'],'left_step':left['step'],'right_episode':right['episode_id'],'right_step':right['step'],'different_public_context':True,'different_known_symbolic_context':symbolic_difference,'scorable':valid,'both_seven_fact_successors_correct':bool(all(li['correct']) and all(ri['correct'])) if valid else None})
scorable=[x for x in pairs if x['scorable']];different=[x for x in scorable if x['different_known_symbolic_context']]
outcomes=[]
for j,name in enumerate(['executor_success','red_moved','blue_moved','pink_moved','led_was_off','led_was_on','light_was_off','light_was_on']):
 ps=np.clip([r['outcome_probabilities'][j] for r in rows],1e-7,1-1e-7);ys=np.asarray([r['actual_outcomes'][j] for r in rows]);pred=ps>=.5
 outcomes.append({'name':name,'facts':len(rows),'NLL':float(-(ys*np.log(ps)+(1-ys)*np.log(1-ps)).mean()),'accuracy':float((pred==ys).mean()),'positive_F1':float(2*((pred==1)&(ys==1)).sum()/max(pred.sum()+ys.sum(),1)),'actual_positive_rate':float(ys.mean())})
result={'checkpoint_sha256':analysis['checkpoint_sha256'],'split':analysis['split'],'roots':analysis['roots'],'transitions':len(rows),'predicate_table':table,'whole_state_accuracy':joint,'outcome_table':outcomes,'matched_context_pairs':len(pairs),'scorable_pairs':len(scorable),'both_contexts_correct_rate':sum(x['both_seven_fact_successors_correct'] for x in scorable)/len(scorable) if scorable else None,'distinct_symbolic_context_pairs':len(different),'both_distinct_symbolic_contexts_correct_rate':sum(x['both_seven_fact_successors_correct'] for x in different)/len(different) if different else None,'definitions':{'NLL':'Negative log probability of valid offline label; lower is better','whole_state_accuracy':'All seven categorical/binary variables correct simultaneously; excludes boundaries with any masked evaluator label and reports excluded count','unknown':'Diagnostic max class confidence below0.5; calibrated rejection is reported separately','paired_context':'Earliest occurrence of same skill in two different branches of one root, with different actual RGB or robot input; both real-history one-step predictions must be wholly correct','conditional_accuracy':'Raw changes and unchanged cases are both retained; no failed skills removed'},'test_data_used':False}
(args.input/'offline_summary.json').write_text(json.dumps(result,indent=2));(args.input/'matched_context_pairs.json').write_text(json.dumps(pairs,indent=2))
with (args.input/'predicate_metrics.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
print('PREDICTION_AGGREGATION_COMPLETE',len(rows),len(pairs))
