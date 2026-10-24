from common import *
def main():
 assert not(E/'results/final_verification.json').exists();summary=read(E/'results/summary.json')['rows'];points=read(E/'results/fixed_threshold_points.json');rows=read(E/'data/ABC_labels_metadata.json');checks=read(E/'results/access_and_checks.json');assert checks['D_file_reads']==0 and checks['old_writes']==0 and checks['training_updates']==0
 assert all(sha(p)==h for p,h in read(E/'configs/pre_run_frozen.json')['sha256'].items());curve_count=0
 for q in summary:
  z=np.load(E/f'data/pr_seed{q["seed"]}_{q["split"]}_{q["camera"]}_{q["category"]}.npz');r=z['recall'];p=z['precision'];assert np.all(np.diff(r)>=0)and np.all(np.diff(z['threshold'])<0);assert np.all((r>=0)&(r<=1))and np.all((p>=0)&(p<=1));assert int(z['TP'][-1])+q['never_matched_GT']==q['GT'];assert int(z['TP'][-1]+z['FP'][-1])==q['kept_candidates'];envelope=np.maximum.accumulate(p[::-1])[::-1];dr=np.diff(np.r_[0.,r]);assert abs(float(np.sum(dr*envelope))-q['AP50_all_points'])<1e-12;assert sum(q['TP_hist_counts'])+q['never_matched_GT']==q['GT'];assert sum(q['TP_hist_counts'])+sum(q['FP_hist_counts'])==q['kept_candidates'];curve_count+=1
 for seed in [0,1]:
  for split in ['train','dev']:
   a=np.load(E/f'data/candidates_seed{seed}_{split}.npz');assert set(a['image_id'])<=set(r['image_id']for r in rows if r['split']==split)
   for c in [0,1]:
    sel=a['class_id']==c
    for i in np.unique(a['image_id'][sel]):
     mask=sel&(a['image_id']==i);gt=a['matched_gt'][mask][a['tp'][mask]==1];assert len(gt)==len(set(gt.tolist()));assert rows[int(i)]['class_valid'][c]
   for q in points:
    if q['seed']!=seed or q['split']!=split:continue
    sel=a['score']>=q['threshold']
    if q['camera']!='all':sel&=a['camera_id']==['static','gripper'].index(q['camera'])
    if q['category']!='all':sel&=a['class_id']==CATS.index(q['category'])
    assert int(a['tp'][sel].sum())==q['TP']and int(sel.sum())==q['TP']+q['FP'];assert q['TP']+q['FN']==q['GT']
 save(E/'results/final_verification.json',{'utc':now(),'passed':True,'curves_verified':curve_count,'fixed_threshold_rows_verified':len(points),'one_to_one_and_unknown_exclusion_verified':True,'AP_independent_precision_envelope_sum_verified':True,'all_GT_denominators_verified':True,'pre_run_frozen_code_and_protocol_unchanged':True,'verification_reads_only_new_ABC_diagnostic_files':True,'D_reads':0,'extra_training_or_inference':0});print('PR_VERIFIED',curve_count,len(points),flush=True)
if __name__=='__main__':main()
