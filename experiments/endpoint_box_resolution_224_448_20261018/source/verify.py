from common import *
import collections,shutil,csv

def overlap(a,b):
 w=max(0.,min(a[2],b[2])-max(a[0],b[0]));h=max(0.,min(a[3],b[3])-max(a[1],b[1]));inter=w*h
 union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
 return inter/union if union>0 else 0.
def main():
 assert not(E/'results/verification.json').exists()
 for name in ['pre_execution_frozen','source_references','training_frozen','final_heads_frozen']:check_manifest(E/f'configs/{name}.json')
 old={}
 for name in ['endpoint_box_localizer_20261018','visual_node_edge_readout_20261018','visual_edge_zero_context_control_20261018','endpoint_rgb_feasibility_20261018']:
  m=P/f'experiments/{name}/configs/delivery_manifest.json';old[name]={'manifest_sha256':sha(m),'files':check_manifest(m)}
 e=read(E/'results/evaluation.json');known=0;boxes=0;metrics=0
 for r in e['records']:
  assert all(p['score']>=.5 for p in r['predictions'])
  if r['status']=='uncertain':assert r['TP']is None and r['FP']is None and r['FN']is None;continue
  gt=r['GT_boxes'];rem=list(range(len(gt)));tp=0
  for p in sorted(r['predictions'],key=lambda p:-p['score']):
   if rem:
    vals=[overlap(p['box'],gt[k])for k in rem];i=int(np.argmax(vals))
    if vals[i]>=.5:tp+=1;rem.pop(i)
  assert(tp,len(r['predictions'])-tp,len(gt)-tp)==(r['TP'],r['FP'],r['FN'])
  best=[max((overlap(p['box'],g)for p in r['predictions']),default=0)for g in gt];assert np.allclose(best,r['best_IoU_per_GT'],atol=1e-8)
  known+=1;boxes+=len(gt);metrics+=3
 for r in e['groups']:
  dims=r['dimensions'].split('/');rr=[x for x in e['records']if x['resolution']==r['resolution']and x['seed']==r['seed']and all(x[d]==r[d]for d in dims)];kk=[x for x in rr if x['status']!='uncertain'];assert len(rr)==r['class_images']and len(kk)==r['known_class_images']
  for k in ['TP','FP','FN']:assert sum(x[k]for x in kk)==r[k]
  ii=[i for x in kk for i in x['best_IoU_per_GT']]
  assert r['mean_best_IoU_including_misses']is None if not ii else abs(np.mean(ii)-r['mean_best_IoU_including_misses'])<1e-9
  assert r['not_visible_FP_image_rate']is None
 assert(len(e['records']),known,boxes,metrics)==(320,224,364,672)
 rows=read(E/'data/public/images.json');anns=read(E/'data/annotations/final.json')['annotations']
 assert sha(E/'data/annotations/final.json')==sha(BASE/'data/annotations/final.json')
 for res in(224,448):
  t=np.load(E/f'data/features/r{res}.npz')['tokens'];ix=[r['image_id']for r in rows if r['split']=='train'];norm=np.load(E/f'configs/feature_normalization_r{res}.npz')
  assert np.array_equal(norm['mean'],t[ix].mean((0,1)))and np.array_equal(norm['std'],np.maximum(t[ix].std((0,1)),.05))
 attempts=read(E/'results/training_attempts.json');assert len(attempts)==2 and all(a['status']=='completed'and a['resolution']==448 for a in attempts)
 training=read(E/'results/training.json');assert all(x['updates']==600 for x in training['runs'])
 audit=read(E/'results/GT_center_audit.json');paired=[];keys={}
 for r in audit:keys[r['resolution'],r['seed'],r['image_id'],r['category'],r['GT_index']]=r
 assert len(keys)==364
 for r in audit:
  if r['resolution']!=224:continue
  other=keys[448,r['seed'],r['image_id'],r['category'],r['GT_index']]
  paired.append({k:r[k]for k in ['seed','image_id','root_id','split','scene','camera','category','GT_index','short_side_patch224']}|{'matched224':r['matched'],'matched448':other['matched'],'best_IoU224':r['best_IoU'],'best_IoU448':other['best_IoU'],'transition':'retained'if r['matched']and other['matched']else'lost'if r['matched']else'gained'if other['matched']else'missed_both'})
 strata=[]
 for res in(224,448):
  for seed in(0,1):
   for split in SPLITS:
    for camera in(0,1):
     for cat in CATS:
      rr=[r for r in audit if r['resolution']==res and r['seed']==seed and r['split']==split and r['camera']==camera and r['category']==cat]
      if not rr:continue
      for size in ['all','short_side_less_than1patch224','short_side_at_least1patch224']:
       z=[r for r in rr if size=='all'or(r['short_side_patch224']<1 if size=='short_side_less_than1patch224'else r['short_side_patch224']>=1)]
       if not z:continue
       strata.append({'resolution':res,'seed':seed,'split':split,'camera':camera,'category':cat,'size':size,'GT_boxes':len(z),'TP':sum(x['matched']for x in z),'mean_best_IoU':float(np.mean([x['best_IoU']for x in z])),'miss_center_causes':dict(collections.Counter(x['first_center_gate_failure']for x in z if not x['matched']))})
 save(E/'results/paired_GT_comparison.json',paired);save(E/'results/stratified_diagnostic.json',strata)
 for name,data in [('paired_GT_comparison.csv',paired),('stratified_diagnostic.csv',strata)]:
  with(R/name).open('w')as f:
   w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 save(E/'results/verification.json',{'utc':now(),'passed':True,'class_image_records':320,'known_class_image_records':known,'independently_recounted_TP_FP_FN_cells':metrics,'GT_scores':boxes,'group_sets':len(e['groups']),'old224_prediction_and_metric_exact_reproduction':e['224_predictions_and_metrics_exactly_reproduced'],'train_only_normalization':True,'same_frozen_annotations':True,'unknowns_unscored':True,'not_visible_examples':0,'new_fits':2,'reused224_fits':2,'updates_each':600,'previous_seals_unchanged':old,'no_DINO_T_SupCon_updates':True,'no_extra_resolution_or_network':True})
 shutil.copy2(__file__,E/'source/verify.py')
 save(E/'configs/evaluation_frozen.json',{'utc':now(),'sha256':{str(p):sha(p)for p in(E/'results').rglob('*')if p.is_file()}})
 print('RESOLUTION_INDEPENDENT_VERIFICATION_PASSED',metrics,boxes,flush=True)
if __name__=='__main__':main()
