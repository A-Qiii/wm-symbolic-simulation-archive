from common import *
from pr_engine import pr
from evaluate import boot_ap

def main():
 assert not(E/'results/final_verification.json').exists();protocol=read(E/'configs/protocol.json');opened=install_guard(protocol['old_inputs_allowlist']);assert all(sha(p)==h for p,h in read(E/'configs/input_frozen.json')['sha256'].items());assert all(sha(p)==h for p,h in read(E/'configs/precollection_frozen.json')['sha256'].items());assert all(sha(p)==h for p,h in read(E/'configs/evaluation_code_frozen.json')['sha256'].items());assert all(sha(p)==h for p,h in read(E/'configs/data_frozen.json')['sha256'].items());assert all(sha(p)==h for p,h in read(E/'results/predictions_frozen.json')['sha256'].items());rows=read(E/'data/public/images.json');ann=read(E/'data/annotation_private/labels.json');metrics=read(E/'results/evaluation_summary.json')['rows'];roots=read(E/'results/root_metrics.json');cal=read(E/'configs/calibration_locked.json')['rows'];rootnames=[q['root_id']for q in protocol['roots']];weights=np.load(E/'data/evaluation/bootstrap_root_multiplicities.npy');assert weights.shape==(2000,16)and np.all(weights.sum(1)==16);n=0
 for seed in [0,1]:
  assert sha(E/f'models/final_seed{seed}.pt')==sha(BASE/f'models/final_seed{seed}.pt');a=dict(np.load(E/f'data/evaluation/candidates_seed{seed}.npz'));thresholds=np.array([next(q['threshold']if q['threshold']is not None else np.inf for q in cal if q['seed']==seed and q['category']==c)for c in CATS])
  for cid in [0,1]:
   for i in np.unique(a['image_id'][a['class_id']==cid]):
    assert ann[int(i)]['class_valid'][cid];sel=(a['image_id']==i)&(a['class_id']==cid)&(a['tp']==1);m=a['matched_gt'][sel];assert len(m)==len(set(m.tolist()))
  for q in [q for q in metrics if q['seed']==seed]:
   sel=np.ones(len(a['score']),bool)
   if q['camera']!='all':sel &= a['camera_id']==['static','gripper'].index(q['camera'])
   if q['category']!='all':sel &= a['class_id']==CATS.index(q['category'])
   b={k:v[sel]for k,v in a.items()};op=b['score']>=(.5 if q['workpoint']=='fixed0.5'else thresholds[b['class_id']]);assert q['TP']==int(b['tp'][op].sum())and q['FP']==int(op.sum())-q['TP']and q['FN']==q['GT']-q['TP'];rr=[x for x in roots if all(q[k]==x[k]for k in ['seed','camera','category','workpoint'])];assert len(rr)==16 and all(sum(r[k]for r in rr)==q[k]for k in ['TP','FP','FN','GT'])
   if q['workpoint']=='calibrated':continue
   gt=np.array([next(x['GT']for x in rr if x['root_id']==r)for r in rootnames]);curve=np.load(E/f'data/evaluation/pr_seed{seed}_{q["camera"]}_{q["category"]}.npz');assert abs(float(np.sum(np.diff(np.r_[0,curve['recall']])*curve['precision_envelope']))-q['AP50'])<1e-12
   for rep in [0,17,1999]:
    w=weights[rep];ix=np.repeat(np.arange(len(b['score'])),w[b['root_idx']]);_,ap=pr(b['score'][ix],b['tp'][ix],np.arange(len(ix)),b['patch'][ix],b['slot'][ix],int(w@gt));compact=boot_ap(b,gt,w[None])[0];assert abs(ap-compact)<1e-12;n+=1
 for q in cal:
  a=np.load(E/f'data/calibration/candidates_seed{q["seed"]}_dev.npz');cid=CATS.index(q['category']);sel=a['class_id']==cid;s=a['score'][sel];tp=a['tp'][sel];order=np.argsort(-s,kind='stable');s,tp=s[order],tp[order];ends=np.r_[np.flatnonzero(s[:-1]!=s[1:]),len(s)-1];ct=np.cumsum(tp)[ends];cn=ends+1;ok=np.where(ct/cn>=.9)[0]
  if len(ok):i=max(ok,key=lambda j:(int(ct[j]),float(s[ends[j]])));assert q['threshold']==float(s[ends[i]])and q['TP']==ct[i]
  else:assert q['threshold']is None
 save(E/'results/final_verification.json',{'utc':now(),'passed':True,'old_input_files_unchanged':len(read(E/'configs/input_frozen.json')['sha256']),'data_and_prediction_hashes_unchanged':True,'model_and_calibration_locked_before_collection':True,'old_C_calibration_optimum_independently_recomputed':4,'normal_metric_rows_verified':len(metrics),'root_metric_rows_verified':len(roots),'expanded_root_bootstrap_AP_checks':n,'same_root_pairs_across_conditions':True,'AP_identical_across_workpoints_by_definition':True,'new_fits':0,'old_D_holdout_sample_feature_prediction_metric_reads':0,'old_files_opened':sorted(opened),'no_old_artifact_writes':True});print('FRESH_C16_VERIFIED',len(metrics),len(roots),n,flush=True)
if __name__=='__main__':main()
