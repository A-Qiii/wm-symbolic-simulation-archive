from common import *
import shutil,math,collections
PREVIOUS={'endpoint_box_decoder_all_candidates_20261018':'49126c3a1a6123eea942101e5a9807fd984266f7604754cea0d19358ecfd91cd','endpoint_box_resolution_224_448_20261018':'21676d8ff1b435a9b4bc8fd10b427cd29884530516d3659351f7859eca4f28c1','endpoint_box_localizer_20261018':'63a7cfa49e9ec31fcdf068f6883e61a1eae625b4b61c3a5602f888a4412fdfa6','visual_node_edge_readout_20261018':'76d6544fe5ae91b3c7880763bd7e7441ef0381a96b7ab2fa0cebc7e8d4027aa7','visual_edge_zero_context_control_20261018':'bf372b6744f5e355e851dcc17cfdbbfd0e7efc8073d2cb01d9231433cf79e2a1','endpoint_rgb_feasibility_20261018':'eacf21686aea5dc469f9a7802dd86546f2a585c6b2e37441bfafd15440c31bc1'}
def independent_match(gt,pp):
 g=np.array(gt,dtype=np.float64).reshape(-1,4);p=np.array([x['box']for x in pp],dtype=np.float64).reshape(-1,4)
 delta=np.maximum(0,np.minimum(p[:,None,2:],g[None,:,2:])-np.maximum(p[:,None,:2],g[None,:,:2]));inter=np.prod(delta,axis=-1);union=np.prod(p[:,2:]-p[:,:2],axis=-1)[:,None]+np.prod(g[:,2:]-g[:,:2],axis=-1)[None,:]-inter;matrix=inter/np.maximum(union,1e-12)
 available=np.ones(len(g),dtype=bool);pairs=[]
 for j in range(len(p)):
  scores=matrix[j].copy();scores[~available]=-1
  if len(g)and scores.max()>=.5:
   k=int(scores.argmax());available[k]=False;pairs.append((j,k,float(matrix[j,k])))
 return len(pairs),len(p)-len(pairs),len(g)-len(pairs),matrix.max(axis=0).tolist()if len(p)else[0.]*len(g),pairs

def main():
 assert not(E/'results/verification.json').exists()
 for name in['pre_adjudication_frozen','labels_frozen','sensitivity_frozen','source_references']:check_manifest(E/f'configs/{name}.json')
 d=read(E/'results/sensitivity.json');assert len(d['records'])==512
 source=read(P/'experiments/endpoint_box_decoder_all_candidates_20261018/results/evaluation.json');old={(r['resolution'],r['seed'],r['variant'],r['image_id'],r['category']):r for r in source['records']if r['image_id']in IDS};assert len(old)==128
 changes=[];matched=0;repro=0;pair_index={}
 for r in d['records']:
  assert r['image_id']in IDS
  pair_index[r['resolution'],r['seed'],r['variant'],r['label_version'],r['support'],r['image_id'],r['category']]=r
  if r['eligible']:
   tp,fp,fn,ivals,pairs=independent_match(r['GT_boxes'],r['predictions']);assert(tp,fp,fn)==(r['TP'],r['FP'],r['FN']);assert np.allclose(ivals,r['best_IoU_per_GT'],atol=1e-14,rtol=0)
   assert [(p,g)for p,g,_ in pairs]==[(x['prediction'],x['GT'])for x in r['matches']];matched+=1
  else:assert r['TP']is None and r['FP']is None and r['FN']is None
  if r['label_version']=='original'and r['support']=='all_known_within_selected8':
   ref=old[r['resolution'],r['seed'],r['variant'],r['image_id'],r['category']]
   for k in ['TP','FP','FN','predictions','matches','best_IoU_per_GT']:assert r[k]==ref[k],(r['image_id'],k)
   repro+=1
 for r in d['records']:
  if r['label_version']!='original'or r['support']!='all_known_within_selected8':continue
  n=pair_index[r['resolution'],r['seed'],r['variant'],'adjudicated',r['support'],r['image_id'],r['category']]
  changes.append({k:r[k]for k in ['resolution','seed','variant','image_id','split','camera','category']}|{'coverage_change':'retained'if r['eligible']and n['eligible']else'added'if n['eligible']else'dropped'if r['eligible']else'remains_unknown','old_TP':r['TP'],'new_TP':n['TP'],'old_FP':r['FP'],'new_FP':n['FP'],'old_FN':r['FN'],'new_FN':n['FN'],'old_GT_count':len(r['GT_boxes']),'new_GT_count':len(n['GT_boxes']),'old_mean_best_IoU':float(np.mean(r['best_IoU_per_GT']))if r['best_IoU_per_GT']else None,'new_mean_best_IoU':float(np.mean(n['best_IoU_per_GT']))if n['best_IoU_per_GT']else None})
 for g in d['groups']:
  rr=[r for r in d['records']if all(r[k]==g[k]for k in ['resolution','seed','variant','label_version','support'])and(g['split']=='selected8_all'or r['split']==g['split'])and(g['category']=='all_categories'or r['category']==g['category'])and(g['camera']=='all'or r['camera']==g['camera'])];kk=[r for r in rr if r['eligible']]
  assert g['class_images']==len(rr)and g['scored_class_images']==len(kk)
  assert(g['TP'],g['FP'],g['FN'],g['GT_boxes'])==(sum(r['TP']for r in kk),sum(r['FP']for r in kk),sum(r['FN']for r in kk),sum(len(r['GT_boxes'])for r in kk))
  if g['split']=='AB_episode_heldout_development':assert not rr and g['recall']is None
 seals={}
 for name,h in PREVIOUS.items():
  path=P/'experiments'/name/'configs/delivery_manifest.json';assert sha(path)==h,(name,sha(path));check_manifest(path);seals[name]={'sha256':h,'files_verified':read(path)['files']}
 assert not list(E.rglob('*.pt'))and not(E/'data/features').exists()
 assert d['new_forward_passes']==d['new_training_steps']==d['new_features']==d['new_inputs']==0 and not d['decoder_rerun']and not d['full40_metrics_recomputed']
 shutil.copy2(__file__,E/'source/verify.py');save(E/'results/per_class_image_metric_changes.json',changes)
 save(E/'results/verification.json',{'utc':now(),'passed':True,'records':len(d['records']),'scored_records_independently_matched':matched,'old_selected_records_reproduced_exactly':repro,'aggregate_rows_verified':len(d['groups']),'zero_AB_development_support_verified':True,'no_labels_changed_after_freeze':True,'previous_seals_unchanged':seals,'all_selected_raw_pixels_source_hashes_verified':True,'saved_prediction_source_sha256':d['prediction_source_sha256'],'no_forward_training_feature_extraction_or_decoding':True,'scope':'No full-data metrics recomputation; prior artifacts hashed only for preservation'})
 print('EIGHT_IMAGE_VERIFIED',matched,'independent matching rows',repro,'old rows exact',len(d['groups']),'aggregates;',sum(x['files_verified']for x in seals.values()),'old files unchanged')
 print('CHANGED_RETAINED_MATCHES',json.dumps([x for x in changes if x['coverage_change']=='retained'and any(x['old_'+k]!=x['new_'+k]for k in ['TP','FP','FN'])],ensure_ascii=False))
 print('NEWLY_INCLUDED_ROWS',json.dumps([x for x in changes if x['coverage_change']=='added'and x['variant']=='corrected'],ensure_ascii=False))
if __name__=='__main__':main()
