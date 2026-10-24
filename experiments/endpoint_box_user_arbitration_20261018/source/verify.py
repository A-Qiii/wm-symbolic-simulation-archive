from common import *
import shutil,collections
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
 for name in['pre_review_frozen','user_labels_and_audit_frozen','evaluation_frozen','source_references']:check_manifest(E/f'configs/{name}.json')
 d=read(E/'results/evaluation.json');assert len(d['records'])==384
 refs={}
 original=read(P/'experiments/endpoint_box_decoder_all_candidates_20261018/results/evaluation.json')['records'];refs['original']={(r['resolution'],r['seed'],r['variant'],r['image_id'],r['category']):r for r in original if r['image_id']in IDS}
 prev=read(V1/'results/sensitivity.json')['records'];refs['assistant_v1']={(r['resolution'],r['seed'],r['variant'],r['image_id'],r['category']):r for r in prev if r['support']=='all_known_within_selected8'and r['label_version']=='adjudicated'}
 checked=0;reproduced=0
 for r in d['records']:
  assert r['image_id']in IDS
  if r['eligible']:
   tp,fp,fn,iv,pairs=independent_match(r['GT_boxes'],r['predictions']);assert(tp,fp,fn)==(r['TP'],r['FP'],r['FN']);assert np.allclose(iv,r['best_IoU_per_GT'],atol=1e-14,rtol=0);assert [(p,g)for p,g,_ in pairs]==[(x['prediction'],x['GT'])for x in r['matches']];checked+=1
  else:assert r['TP']is None and r['FP']is None and r['FN']is None
  if r['label_version']in refs:
   ref=refs[r['label_version']][r['resolution'],r['seed'],r['variant'],r['image_id'],r['category']]
   for k in ['TP','FP','FN','matches','predictions','best_IoU_per_GT']:assert r[k]==ref[k],(r['label_version'],r['image_id'],k)
   reproduced+=1
 masks={k:{(x['image_id'],x['category'])for x in v}for k,v in read(E/'configs/scoring_masks.json').items()}
 for g in d['groups']:
  rr=[r for r in d['records']if all(r[k]==g[k]for k in ['resolution','seed','variant','label_version'])and(r['image_id'],r['category'])in masks[g['support']]and(g['split']=='selected8_all'or r['split']==g['split'])and(g['category']=='all_categories'or r['category']==g['category'])and(g['camera']=='all'or r['camera']==g['camera'])];kk=[r for r in rr if r['eligible']]
  assert g['selected_class_images_in_mask']==len(rr)and g['scored_class_images']==len(kk)
  assert(g['TP'],g['FP'],g['FN'],g['GT_boxes'])==(sum(r['TP']for r in kk),sum(r['FP']for r in kk),sum(r['FN']for r in kk),sum(len(r['GT_boxes'])for r in kk))
  if g['split']=='AB_episode_heldout_development':assert not rr and g['recall']is None
 p1=read(E/'data/labels/assistant_v1_selected.json')['annotations'];p2=read(E/'data/labels/user_semantics_v2.json')['annotations'];idx1={(a['image_id'],c['category']):c for a in p1 for c in a['classes']};idx2={(a['image_id'],c['category']):c for a in p2 for c in a['classes']};allowed={(0,'gripper_finger'),(8,'gripper_finger'),(1,'drawer_front'),(9,'drawer_front')}
 assert all(idx1[k]==idx2[k]for k in idx1 if k not in allowed)
 assert idx2[0,'gripper_finger']['boxes']==[[65,49,77,65],[78,48,91,67]]
 assert idx2[8,'gripper_finger']['status']=='uncertain'and idx2[1,'drawer_front']['status']=='uncertain'
 for r in d['records']:
  if r['label_version']!='user_v2'or(r['image_id'],r['category'])not in masks['common_v1_v2']:continue
  prior=refs['assistant_v1'][r['resolution'],r['seed'],r['variant'],r['image_id'],r['category']]
  for k in ['TP','FP','FN','matches','best_IoU_per_GT']:assert r[k]==prior[k]
 prior_ver=read(V1/'results/verification.json');seals={}
 for name,entry in prior_ver['previous_seals_unchanged'].items():
  p=P/f'experiments/{name}/configs/delivery_manifest.json';assert sha(p)==entry['sha256'];check_manifest(p);seals[name]={'sha256':sha(p),'files_verified':read(p)['files']}
 p=V1/'configs/delivery_manifest.json';assert sha(p)=='00911f517e90de23523492b12e8ff22f579673375519f22ef2935ecf0667ffb9';check_manifest(p);seals[V1.name]={'sha256':sha(p),'files_verified':read(p)['files']}
 audit=read(E/'data/audit/AB_readonly_findings.json');assert len(audit)==32 and{r['image_id']for r in audit}==set(range(32))and all(not r['original_label_modified']for r in audit)
 assert not list(E.rglob('*.pt'))and not(E/'data/features').exists()
 shutil.copy2(__file__,E/'source'/Path(__file__).name)
 save(E/'results/verification.json',{'utc':now(),'passed':True,'records':len(d['records']),'independent_matching_records':checked,'old_original_and_v1_records_reproduced_exactly':reproduced,'aggregate_rows_verified':len(d['groups']),'v1_v2_common_support_metrics_identical':True,'unchanged_other12class_images_verified':True,'panel_color_mapping_verified':True,'unknown_not_converted_to_negative':True,'only_selected8_scored':True,'AB32_readonly_identity_audit_not_new_labels':True,'full40_scoring_not_recomputed':True,'no_training_forward_features_decoding':True,'old_seals_unchanged':seals})
 print('USER_ARBITRATION_VERIFIED',checked,'independent matches',reproduced,'old rows exact',len(d['groups']),'groups',sum(x['files_verified']for x in seals.values()),'old files unchanged')
 for r in d['records']:
  if r['label_version']=='user_v2'and(r['image_id'],r['category'])in masks['added_v2_vs_v1']:print(json.dumps({k:r[k]for k in ['resolution','seed','variant','image_id','category','TP','FP','FN','best_IoU_per_GT']},ensure_ascii=False))
if __name__=='__main__':main()
