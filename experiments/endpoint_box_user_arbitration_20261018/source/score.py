from common import *
import csv,shutil
DEC=P/'experiments/endpoint_box_decoder_all_candidates_20261018'
SPLITS=['train','AB_episode_heldout_development','C_seen_diagnostic','D_seen_diagnostic']
def iou(a,b):
 inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]));return inter/max((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter,1e-12)
def match(pp,gt):
 remaining=set(range(len(gt)));matches=[]
 for j,p in enumerate(pp):
  k=max(remaining,key=lambda k:iou(p['box'],gt[k]))if remaining else None
  if k is not None and iou(p['box'],gt[k])>=.5:remaining.remove(k);matches.append({'prediction':j,'GT':k,'IoU':iou(p['box'],gt[k])})
 return {'TP':len(matches),'FP':len(pp)-len(matches),'FN':len(gt)-len(matches),'matches':matches,'best_IoU_per_GT':[max((iou(p['box'],g)for p in pp),default=0)for g in gt]}
def csvsave(p,rr):
 keys=list(dict.fromkeys(k for r in rr for k in r))
 with Path(p).open('w')as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rr)
def aggregate(rr):
 kk=[r for r in rr if r['eligible']];unknown=[r for r in rr if not r['eligible']];tp=sum(r['TP']for r in kk);fp=sum(r['FP']for r in kk);fn=sum(r['FN']for r in kk);vs=[v for r in kk for v in r['best_IoU_per_GT']]
 return {'selected_class_images_in_mask':len(rr),'scored_class_images':len(kk),'unknown_class_images':len(unknown),'GT_boxes':tp+fn,'TP':tp,'FP':fp,'FN':fn,'precision':tp/(tp+fp)if tp+fp else None,'recall':tp/(tp+fn)if tp+fn else None,'mean_best_IoU_including_misses':sum(vs)/len(vs)if vs else None,'unknown_predictions_unscored':sum(r['pred_count']for r in unknown)}
def main():
 check_manifest(E/'configs/user_labels_and_audit_frozen.json');check_manifest(E/'configs/pre_review_frozen.json');assert not(E/'results').exists()
 labels={v:{(a['image_id'],c['category']):c for a in read(E/p)['annotations']for c in a['classes']}for v,p in [('original','data/labels/original_selected.json'),('assistant_v1','data/labels/assistant_v1_selected.json'),('user_v2','data/labels/user_semantics_v2.json')]}
 known={v:{k for k,a in ld.items()if a['status']!='uncertain'}for v,ld in labels.items()}
 masks={'all_known_selected8':set(labels['original']),'common_v1_v2':known['assistant_v1']&known['user_v2'],'added_v2_vs_v1':known['user_v2']-known['assistant_v1'],'common_original_v2':known['original']&known['user_v2'],'added_v2_vs_original':known['user_v2']-known['original'],'dropped_original_vs_v2':known['original']-known['user_v2'],'all_three_common':set.intersection(*known.values())}
 assert masks['added_v2_vs_v1']=={(0,'gripper_finger'),(9,'drawer_front')}
 save(E/'configs/scoring_masks.json',{k:[{'image_id':i,'category':c}for i,c in sorted(v)]for k,v in masks.items()})
 den=[];meta={r['image_id']:r for r in read(E/'data/images_selected8.json')}
 for i,c in sorted(labels['original']):
  row={'image_id':i,'category':c,'split':meta[i]['split'],'camera':meta[i]['camera']}
  for v,ld in labels.items():row.update({v+'_status':ld[i,c]['status'],v+'_boxes':len(ld[i,c]['boxes'])})
  row.update({k:(i,c)in v for k,v in masks.items()if k!='all_known_selected8'});den.append(row)
 save(E/'data/labels/denominator_changes.json',den);csvsave(R/'denominator_changes.csv',den)
 source=DEC/'results/decoded_predictions.json';h=sha(source);assert h=='525ee16a677dffb1b1c03e4e6228c4e7cbc4ddff600c4028054f404e47362df3'
 outputs=[p for p in read(source)if p['image_id']in IDS];assert len(outputs)==32;save(E/'results/saved_predictions_selected8.json',outputs)
 records=[]
 for out in outputs:
  i=out['image_id'];r=meta[i]
  for variant in ['original','corrected']:
   for version,ld in labels.items():
    for cat in CATS:
     a=ld[i,cat];gt=[(np.array(b)/[r['width'],r['height'],r['width'],r['height']]).tolist()for b in a['boxes']];pp=[p for p in out[variant]if p['category']==cat];assert all(p['score']>=.5 for p in pp)and all(pp[j]['score']>=pp[j+1]['score']for j in range(len(pp)-1))
     rr={k:r[k]for k in ['image_id','root_id','state_id','scene','split','camera']};rr.update(resolution=out['resolution'],seed=out['seed'],variant=variant,label_version=version,category=cat,status=a['status'],eligible=a['status']!='uncertain',GT_boxes=gt,predictions=pp,pred_count=len(pp));rr.update(match(pp,gt)if rr['eligible']else{'TP':None,'FP':None,'FN':None,'matches':[],'best_IoU_per_GT':[]});records.append(rr)
 groups=[]
 for res in[224,448]:
  for seed in[0,1]:
   for variant in ['original','corrected']:
    for version in labels:
     base=[r for r in records if(r['resolution'],r['seed'],r['variant'],r['label_version'])==(res,seed,variant,version)]
     for mask,keys in masks.items():
      for split in ['selected8_all']+SPLITS:
       for cat in ['all_categories']+CATS:
        for cam in(['all',0,1]if mask=='all_known_selected8'else['all']):
         rr=[r for r in base if(r['image_id'],r['category'])in keys and(split=='selected8_all'or r['split']==split)and(cat=='all_categories'or r['category']==cat)and(cam=='all'or r['camera']==cam)]
         groups.append({'resolution':res,'seed':seed,'variant':variant,'label_version':version,'support':mask,'split':split,'category':cat,'camera':cam,**aggregate(rr)})
 save(E/'results/evaluation.json',{'utc':now(),'selected_ids':IDS,'records':records,'groups':groups,'prediction_source':str(source),'prediction_source_sha256':h,'thresholds':{'score':.5,'saved_NMS_IoU':.3,'matching_IoU':.5},'no_training':True,'no_forward':True,'no_feature_extraction':True,'no_decoder_rerun':True,'no_AB_development_scoring':True,'no_full40_recompute':True})
 csvsave(R/'all_group_metrics.csv',groups);csvsave(R/'all_image_metrics.csv',records);shutil.copy2(__file__,E/'source'/Path(__file__).name)
 save(E/'configs/evaluation_frozen.json',{'utc':now(),'sha256':{str(p):sha(p)for p in [E/'results/evaluation.json',E/'results/saved_predictions_selected8.json',E/'configs/scoring_masks.json',E/'data/labels/denominator_changes.json',E/'source/score.py']}})
 print('USER_LABEL_SAVED_PREDICTION_EVALUATION_COMPLETE',len(records),len(groups))
 for g in groups:
  if g['label_version']=='user_v2'and g['split']=='selected8_all'and g['category']=='all_categories'and g['camera']=='all'and g['support']in ['all_known_selected8','common_v1_v2','added_v2_vs_v1']:print(json.dumps({k:g[k]for k in ['resolution','seed','variant','support','scored_class_images','GT_boxes','TP','FP','FN']}))
if __name__=='__main__':main()
