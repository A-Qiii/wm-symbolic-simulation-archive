from pathlib import Path
import json
import numpy as np
P=Path('/home/robot/research/projects/wm_symbolic');O=P/'experiments/state_predicates_20261014';E=P/'experiments/state_predicates_augmented_20261014'
def main():
 e=json.loads((E/'results/evaluation.json').read_text());oldmeta=[r for r in json.loads((O/'data/metadata.json').read_text()) if r['split']=='diagnostic_D'];nm=json.loads((E/'data/metadata_test.json').read_text());regressions=[];inerrors=[]
 with np.load(O/'data/inputs_targets.npz') as z:
  sel=z['splits']=='diagnostic_D';held=z['cat'][sel,:3]==6;objs=z['objects'][sel];oldvalid=z['valid'][sel,:3]
 for seed in [0,1,2]:
  a=np.load(E/'results'/f'old_seed{seed}_old_D_regression_probabilities.npy');b=np.load(E/'results'/f'augmented_seed{seed}_old_D_regression_probabilities.npy')
  for i,j in np.argwhere(held&oldvalid&(a[:,:3]>=.5)&(b[:,:3]<.5)):
   r=oldmeta[i];features=objs[i,-1,j].reshape(2,390);regressions.append({'seed':seed,'index':r['index'],'root_id':r['root_id'],'episode_id':r['episode_id'],'step':r['step'],'object':['red','blue','pink'][j],'predicate_column':int(j),'truth':True,'old_probability':float(a[i,j]),'new_probability':float(b[i,j]),'drawer_category':r['category'][3],'visible_latest_views':features[:,388].astype(bool).tolist(),'input_files':r['public_inputs'],'original_category_valid':bool(oldvalid[i,j])})
  a=np.load(E/'results'/f'old_seed{seed}_new_test_probabilities.npy');b=np.load(E/'results'/f'augmented_seed{seed}_new_test_probabilities.npy')
  for i,r in enumerate(nm):
   j=r['target_index']+3
   if a[i,j]<.5 and b[i,j]>=.5:inerrors.append({'seed':seed,'sample_id':r['sample_id'],'root_id':r['root_id'],'predicate_column':j,'truth':False,'old_probability':float(a[i,j]),'new_probability':float(b[i,j]),'input_files':r['public_frames'],'condition':r['condition']})
 ledger={}
 for seed in [0,1,2]:
  batches=np.load(E/'logs'/f'seed{seed}_batch_indices.npy');assert batches.shape==(1500,60);counts=np.bincount(batches.ravel(),minlength=973);assert counts.sum()==90000
  selected=json.loads((E/'results/training_runs.json').read_text())[seed]['selected_update'];prefix=batches[:selected];ledger[str(seed)]={'updates':1500,'full_presentations':int(counts.sum()),'original_presentations':int(counts[:900].sum()),'supplement_presentations':int(counts[900:].sum()),'selected_update':selected,'selected_presentations':int(prefix.size),'selected_supplement_presentations':int((prefix>=900).sum()),'supplement_supervised_in_facts':0}
 first_regression=next(r for r in regressions if r['seed']==0);first_in=next(r for r in inerrors if r['seed']==2)
 d={'new_held_regressions':regressions,'new_in_false_positives':inerrors,'regressed_root_counts':{str(s):len({r['root_id'] for r in regressions if r['seed']==s}) for s in [0,1,2]},'new_held_regressions_unique_physical_states_note':'Old no-action boundaries repeat state within a root; count records and root groups separately.','budget_ledger':ledger,'first_regression_for_visualization':first_regression,'first_in_error_for_visualization':first_in,'findings':[{'evidence':'All selected and final1500 models make zero raw held/open errors on new test; original models make held6/8/6 andopen8/9/8 errors.','interpretation':'Added data helps these newly sampled distributions even at matching final update count.'},{'evidence':'Original old-D near-not-held2 errors improve only selectedseed0; selectedseed1/2 andallfinal models retain2.','interpretation':'New offset-empty-gripper and supported-contact probes do not establish robust correction of all original failed-grasp appearances.'},{'evidence':'New held FN additions arise on2/3/2oldD roots; all original labels valid and targets visible.','interpretation':'Remaining regression is not explained by missing object observations or a changed held standard; narrower negative bias/representation changes are plausible, not isolated causes.'},{'evidence':'Seed2 produces4new in false positives on supplement states; supplementary in labels were masked, andencoder is shared.','interpretation':'An unsupervised predicate can drift on newly covered states despite unchanged old-distribution in results. No ablation isolates a unique cause.'},{'evidence':'Only new6dev roots select checkpoints; selectedupdates300/150/1500 differ.','interpretation':'Checkpoint selection contributes to the trade-off. Final1500 controls are reported; no test-based choice replaces the preselected model.'}],'no_additional_training_or_parameter_change':True,'no_counterfactual_GT_used_as_model_input':True}
 (E/'results/failure_diagnosis.json').write_text(json.dumps(d,ensure_ascii=False,indent=2));print('DIAGNOSIS_SAVED',len(regressions),len(inerrors),flush=True)
if __name__=='__main__':main()
