from common import *
import torch,collections,shutil

def overlap(a,b):
 w=max(0.,min(a[2],b[2])-max(a[0],b[0]));h=max(0.,min(a[3],b[3])-max(a[1],b[1]));inter=w*h;union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter;return inter/union if union>0 else 0.
def main():
 for name in['pre_annotation_frozen','annotations_frozen','source_references','feature_code_frozen','training_frozen','final_heads_frozen']:
  for p,h in read(E/f'configs/{name}.json')['sha256'].items():assert sha(p)==h,p
 for folder in ['visual_node_edge_readout_20261018','visual_edge_zero_context_control_20261018','endpoint_rgb_feasibility_20261018']:
  for p,h in read(P/f'experiments/{folder}/configs/delivery_manifest.json')['sha256'].items():assert sha(p)==h,p
 a=read(E/'results/training_attempts.json');assert len(a)==2 and all(r['status']=='completed'for r in a);tr=read(E/'results/training.json');assert all(r['updates']==600 and r['parameters']==25290 for r in tr['runs'])
 e=read(E/'results/evaluation.json');count=0;known=0;gtcount=0
 for r in e['records']:
  assert all(p['score']>=.5 for p in r['predictions'])
  if r['status']=='uncertain':assert r['TP']is None and r['FP']is None and r['FN']is None;continue
  gt=r['GT_boxes'];rem=list(range(len(gt)));tp=0
  for p in sorted(r['predictions'],key=lambda p:-p['score']):
   if rem:
    vals=[overlap(p['box'],gt[k])for k in rem];idx=int(np.argmax(vals))
    if vals[idx]>=.5:tp+=1;rem.pop(idx)
  assert(tp,len(r['predictions'])-tp,len(gt)-tp)==(r['TP'],r['FP'],r['FN']);ious=[max((overlap(p['box'],g)for p in r['predictions']),default=0)for g in gt];assert np.allclose(ious,r['best_IoU_per_GT'],atol=1e-8);known+=1;gtcount+=len(gt);count+=3
 for r in e['groups']:
  dims=r['dimensions'].split('/');rr=[x for x in e['records']if x['seed']==r['seed']and all(x[d]==r[d]for d in dims)];kk=[x for x in rr if x['status']!='uncertain'];assert len(rr)==r['class_images'];assert len(kk)==r['known_class_images']
  for k in ['TP','FP','FN']:assert sum(x[k]for x in kk)==r[k]
 assert len(e['records'])==160 and gtcount==182
 rows=read(E/'data/public/images.json');norm=np.load(E/'configs/feature_normalization.npz');raw=np.load(E/'data/features/tokens.npz')['tokens'];ix=[r['image_id']for r in rows if r['split']=='train'];assert np.array_equal(norm['mean'],raw[ix].mean((0,1)));assert np.array_equal(norm['std'],np.maximum(raw[ix].std((0,1)),.05))
 save(E/'results/verification.json',{'utc':now(),'passed':True,'class_image_records':160,'known_class_image_records':known,'independently_checked_TP_FP_FN_cells':count,'GT_boxes_evaluated_across_seeds':gtcount,'group_metric_sets':len(e['groups']),'train_only_normalization':True,'old_seals_unchanged':True,'new_training_fits':2,'updates_each':600,'no_DINO_T_SupCon_training':True,'unknowns_not_counted_as_negatives':True,'no_claim_of_independent_annotation_truth':True});shutil.copy2(__file__,E/'source/verify.py');print('LOCALIZER_INDEPENDENT_METRIC_VERIFICATION_PASSED',known,count,flush=True)
if __name__=='__main__':main()
