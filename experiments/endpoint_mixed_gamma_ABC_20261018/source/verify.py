from common import *
from net import tensor_hash
import torch
def main():
 assert not(E/'results/final_verification.json').exists();protocol=read(E/'configs/protocol.json');allowed=protocol['allowlisted_old_inputs'];opened=install_guard(allowed);assert all(sha(p)==h for p,h in read(E/'configs/input_frozen.json')['sha256'].items());assert all(sha(p)==h for p,h in read(E/'configs/pre_run_frozen.json')['sha256'].items());assert all(sha(p)==h for p,h in read(E/'results/predictions_frozen.json')['sha256'].items());rows=read(E/'data/ABC_labels_metadata.json');sums=read(E/'results/evaluation_summary.json')['rows'];points=read(E/'results/fixed_threshold_points.json');base=read(E/'data/baseline/summary.json')['rows'];comp=read(E/'results/comparison.json');curves=0
 for q in sums:
  z=np.load(E/f'data/evaluation/pr_seed{q["seed"]}_{q["split"]}_{q["camera"]}_{q["category"]}.npz');r,p=z['recall'],z['precision'];assert np.all(np.diff(r)>=0)and np.all(np.diff(z['threshold'])<0)
  if len(r):assert abs(float(np.sum(np.diff(np.r_[0.,r])*np.maximum.accumulate(p[::-1])[::-1]))-q['AP50_all_points'])<1e-12;assert int(z['TP'][-1])+q['never_matched_GT']==q['GT']
  else:assert q['AP50_all_points']==0 and q['never_matched_GT']==q['GT']
  assert sum(q['TP_hist_counts'])+q['never_matched_GT']==q['GT'];curves+=1
 for seed in [0,1]:
  fit=read(E/f'results/train_seed{seed}.json');assert len(fit['history'])==1200 and [r['update']for r in fit['history']]==list(range(1,1201));assert fit['positive_exposures_by_slot']==[14250,37425,825];assert fit['sha256']==sha(E/f'models/final_seed{seed}.pt');initial=torch.load(E/f'models/initial_seed{seed}.pt',map_location='cpu',weights_only=True);assert tensor_hash(initial['state'])==initial['initial_hash'];assert sha(E/f'models/initial_seed{seed}.pt')==sha(BASE/f'models/initial_seed{seed}.pt');assert sha(E/f'configs/schedule_seed{seed}.npy')==sha(BASE/f'configs/schedule_seed{seed}.npy')
  for split in ['train','dev']:
   a=np.load(E/f'data/evaluation/candidates_seed{seed}_{split}.npz');assert set(a['image_id'])<=set(q['image_id']for q in rows if q['split']==split)
   for c in [0,1]:
    for i in np.unique(a['image_id'][a['class_id']==c]):
     sel=(a['class_id']==c)&(a['image_id']==i);gt=a['matched_gt'][sel][a['tp'][sel]==1];assert len(gt)==len(set(gt.tolist()));assert rows[int(i)]['class_valid'][c]
   for q in points:
    if q['seed']!=seed or q['split']!=split:continue
    sel=a['score']>=q['threshold']
    if q['camera']!='all':sel&=a['camera_id']==['static','gripper'].index(q['camera'])
    if q['category']!='all':sel&=a['class_id']==CATS.index(q['category'])
    assert int(a['tp'][sel].sum())==q['TP']and int(sel.sum())==q['TP']+q['FP']and q['TP']+q['FN']==q['GT']
 for q in comp:
  b=next(r for r in base if all(q[k]==r[k]for k in ['seed','split','camera','category']));assert b['AP50_all_points']==q['focal_AP50'];assert abs(q['mixed_AP50']-q['focal_AP50']-q['delta_AP50'])<1e-12
 probes=read(E/'results/shared_layer_diagnostics.json');assert probes['optimizer_updates']==0 and probes['all_parameter_hashes_unchanged'];assert probes['same_initial_finger_direct_gradient_equal'];assert len(probes['initial_probes'])==12
 save(E/'results/final_verification.json',{'utc':now(),'passed':True,'selected_old_input_files_unchanged':len(allowed),'new_training_runs':2,'updates_per_run':1200,'same_initial_weights_and_schedules':True,'same_slot_exposures':True,'evaluation_curves_verified':curves,'fixed_threshold_rows_verified':len(points),'same_initial_finger_direct_gradient_equal':True,'AP_and_GT_denominators_verified':True,'unknown_mask_and_one_to_one_matching_verified':True,'prior_results_reused_unchanged':True,'D_samples_features_predictions_metrics_reads':0,'old_files_opened':sorted(opened),'old_artifact_writes':0});print('MIXED_VERIFIED',curves,len(points),flush=True)
if __name__=='__main__':main()
