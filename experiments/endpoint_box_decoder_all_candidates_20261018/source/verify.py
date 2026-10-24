from common import *
import collections,shutil

def overlap(a,b):
 w=max(0.,min(a[2],b[2])-max(a[0],b[0]));h=max(0.,min(a[3],b[3])-max(a[1],b[1]));inter=w*h;union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
 return inter/union if union>0 else 0.
def main():
 assert not(E/'results/verification.json').exists()
 for name in ['pre_decode_frozen','source_references','outputs_frozen_before_scoring']:check_manifest(E/f'configs/{name}.json')
 prior={}
 for name in ['endpoint_box_resolution_224_448_20261018','endpoint_box_localizer_20261018','visual_node_edge_readout_20261018','visual_edge_zero_context_control_20261018','endpoint_rgb_feasibility_20261018']:
  f=P/f'experiments/{name}/configs/delivery_manifest.json';prior[name]={'sha256':sha(f),'files':check_manifest(f)}
 traces=read(E/'results/candidate_trace.json');by=collections.defaultdict(list)
 for x in traces:by[x['resolution'],x['seed'],x['image_id'],x['category']].append(x)
 outputs=read(E/'results/decoded_predictions.json')
 for r in outputs:
  for c in CATS:
   candidates=sorted([p for p in by[r['resolution'],r['seed'],r['image_id'],c]if p['score']>=.5 and p['valid_box']],key=lambda p:(-p['score'],p['patch']))
   kept=[]
   for p in candidates:
    if all(overlap(p['box'],q['box'])<=.3 for q in kept):kept.append(p)
   expected=[{k:p[k]for k in ['category','score','box','patch']}for p in kept]
   assert expected==[p for p in r['corrected']if p['category']==c],('independentNMS',r,c)
 e=read(E/'results/evaluation.json');known=0;boxes=0
 for r in e['records']:
  if r['status']=='uncertain':assert r['TP']is None and r['FP']is None and r['FN']is None;continue
  gt=r['GT_boxes'];rem=list(range(len(gt)));tp=0
  for p in sorted(r['predictions'],key=lambda p:-p['score']):
   if rem:
    vals=[overlap(p['box'],gt[j])for j in rem];i=int(np.argmax(vals))
    if vals[i]>=.5:tp+=1;rem.pop(i)
  assert(tp,len(r['predictions'])-tp,len(gt)-tp)==(r['TP'],r['FP'],r['FN'])
  best=[max((overlap(p['box'],g)for p in r['predictions']),default=0)for g in gt];assert np.allclose(best,r['best_IoU_per_GT'],atol=1e-8);known+=1;boxes+=len(gt)
 for r in e['groups']:
  dims=r['dimensions'].split('/');rr=[x for x in e['records']if all(x[k]==r[k]for k in ['resolution','seed','variant']+dims)];kk=[x for x in rr if x['status']!='uncertain']
  assert len(rr)==r['class_images']and len(kk)==r['known_class_images']
  for k in ['TP','FP','FN']:assert sum(x[k]for x in kk)==r[k]
  ii=[i for x in kk for i in x['best_IoU_per_GT']]
  assert(r['mean_best_IoU_including_misses']is None)if not ii else abs(np.mean(ii)-r['mean_best_IoU_including_misses'])<1e-9
 assert(len(e['records']),known,boxes)==(640,448,728)
 audit=read(E/'results/GT_candidate_audit.json');assert len(audit)==364
 for a in audit:
  gt=next(x for x in e['records']if x['variant']=='original'and all(x[k]==a[k]for k in ['resolution','seed','image_id','category']))['GT_boxes'][a['GT_index']]
  qs=[x for x in by[a['resolution'],a['seed'],a['image_id'],a['category']]if x['valid_box']and overlap(x['box'],gt)>=.5]
  assert len(qs)==a['qualified_candidate_count']
  if a['corrected_miss_cause']=='no_qualified_candidate':assert not qs
  if a['corrected_miss_cause']=='NMS_removed_all_qualified':assert qs and all(x['corrected_disposition']['reason']=='NMS_suppressed'for x in qs)
 # Exact mechanically identified examples, without changing the decoder.
 adjacent=[]
 for a in audit:
  if a['resolution']==224 and a['split']=='train'and a['image_id']in[0,2,6]and not a['original_matched']:
   q=a['GT_center_qualified_candidate'];assert q and q['original']['reason']=='local_peak_removed'and q['corrected']['reason']=='kept'and a['corrected_matched'];adjacent.append({k:a[k]for k in ['seed','image_id','GT_index']}|q)
 assert len(adjacent)==6
 q=next(x for x in traces if x['resolution']==448 and x['seed']==0 and x['image_id']==27 and x['category']=='gripper_finger'and x['patch']==132)
 assert q['original_disposition']['reason']=='cap_not_processed'and q['corrected_disposition']['reason']=='kept'
 assert not list(E.rglob('*.pt'))and not(E/'data/features').exists()and not(E/'results/training_attempts.json').exists()
 save(E/'results/design_issue_evidence.json',{'adjacent_training_centers':adjacent,'reported_cap_candidate':q,'NMS_implementation_independently_reproduced':True,'no_threshold_change':True})
 save(E/'results/verification.json',{'utc':now(),'passed':True,'independent_NMS_image_class_checks':len(outputs)*2,'class_image_records':640,'known_class_image_records':known,'independently_recounted_TP_FP_FN_cells':known*3,'GT_scores':boxes,'posthoc_GT_candidate_audits':len(audit),'group_sets':len(e['groups']),'all_unmatched_retained_predictions_are_FP':True,'uncertain_is_not_negative':True,'no_not_visible_examples':True,'original_predictions_and_metrics_exact':True,'new_training':0,'new_feature_extractions':0,'new_model_forwards':0,'previous_seals_unchanged':prior})
 shutil.copy2(__file__,E/'source/verify.py')
 save(E/'configs/evaluation_frozen.json',{'utc':now(),'sha256':{str(p):sha(p)for p in(E/'results').glob('*.json')}})
 print('DECODER_INDEPENDENT_NMS_AND_METRICS_VERIFIED',len(outputs)*2,known*3,boxes,flush=True)
if __name__=='__main__':main()
