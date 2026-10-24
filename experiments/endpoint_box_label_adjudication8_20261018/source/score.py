from common import *
import collections,csv,shutil,math
DEC=P/'experiments/endpoint_box_decoder_all_candidates_20261018'
SPLITS=['train','AB_episode_heldout_development','C_seen_diagnostic','D_seen_diagnostic']
def iou(a,b):
 x0=max(a[0],b[0]);y0=max(a[1],b[1]);x1=min(a[2],b[2]);y1=min(a[3],b[3]);inter=max(0,x1-x0)*max(0,y1-y0)
 return inter/max((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter,1e-12)
def match(pp,gt):
 rem=set(range(len(gt)));matches=[]
 for j,p in enumerate(pp):
  g=max(rem,key=lambda k:iou(p['box'],gt[k]))if rem else None
  if g is not None and iou(p['box'],gt[g])>=.5:rem.remove(g);matches.append({'prediction':j,'GT':g,'IoU':iou(p['box'],gt[g])})
 return {'TP':len(matches),'FP':len(pp)-len(matches),'FN':len(gt)-len(matches),'matches':matches,'best_IoU_per_GT':[max((iou(p['box'],g)for p in pp),default=0)for g in gt]}
def aggregate(rr):
 eligible=[x for x in rr if x['eligible']];unk=[x for x in rr if x['status']=='uncertain'];tp=sum(x['TP']for x in eligible);fp=sum(x['FP']for x in eligible);fn=sum(x['FN']for x in eligible);vals=[v for x in eligible for v in x['best_IoU_per_GT']]
 return {'class_images':len(rr),'scored_class_images':len(eligible),'unknown_class_images':len(unk),'known_excluded_by_common_mask':sum(x['exclusion']=='outside_common_known_support'for x in rr),'GT_boxes':tp+fn,'TP':tp,'FP':fp,'FN':fn,'precision':tp/(tp+fp)if tp+fp else None,'recall':tp/(tp+fn)if tp+fn else None,'mean_best_IoU_including_misses':sum(vals)/len(vals)if vals else None,'unknown_images_with_predictions':sum(x['pred_count']>0 for x in unk),'unknown_predictions_unscored':sum(x['pred_count']for x in unk)}
def csvsave(path,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with Path(path).open('w')as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def main():
 check_manifest(E/'configs/labels_frozen.json');check_manifest(E/'configs/pre_adjudication_frozen.json');assert not(E/'results').exists()
 m=read(E/'data/images.json');meta={r['image_id']:r for r in m}
 labels={v:{(a['image_id'],c['category']):c for a in read(E/p)['annotations']for c in a['classes']}for v,p in [('original','data/labels/original_selected.json'),('adjudicated','data/labels/adjudicated_eval_v1.json')]}
 known={v:{k for k,c in d.items()if c['status']!='uncertain'}for v,d in labels.items()};common=known['original']&known['adjudicated']
 den=[]
 for k in sorted(set(labels['original'])|set(labels['adjudicated'])):
  i,c=k;o=labels['original'][k];n=labels['adjudicated'][k];ok=k in known['original'];nk=k in known['adjudicated'];den.append({'image_id':i,'split':meta[i]['split'],'camera':meta[i]['camera'],'category':c,'old_status':o['status'],'new_status':n['status'],'old_boxes':len(o['boxes']),'new_boxes':len(n['boxes']),'old_known':ok,'new_known':nk,'common_known':k in common,'coverage_change':'retained'if ok and nk else'added'if nk else'dropped'if ok else'remains_unknown','coordinate_changed':o['boxes']!=n['boxes']})
 save(E/'data/labels/denominator_changes.json',den);csvsave(R/'denominator_changes.csv',den)
 # First saved-model output access in this script occurs strictly after label freeze validation.
 source=DEC/'results/decoded_predictions.json';source_hash=sha(source);assert source_hash=='525ee16a677dffb1b1c03e4e6228c4e7cbc4ddff600c4028054f404e47362df3'
 outputs=[x for x in read(source)if x['image_id']in IDS];assert len(outputs)==32
 assert {(x['resolution'],x['seed'],x['image_id'])for x in outputs}=={(r,s,i)for r in[224,448]for s in[0,1]for i in IDS}
 save(E/'results/saved_predictions_selected8.json',outputs)
 records=[]
 for out in outputs:
  i=out['image_id'];r=meta[i];assert out['split']==r['split']
  for version,ld in labels.items():
   for variant in ['original','corrected']:
    for cat in CATS:
     a=ld[i,cat];gt=[(np.array(b)/[r['width'],r['height'],r['width'],r['height']]).tolist()for b in a['boxes']];pp=[p for p in out[variant]if p['category']==cat]
     assert all(p['score']>=.5 for p in pp)
     assert all(pp[j]['score']>=pp[j+1]['score']for j in range(len(pp)-1))
     for support in ['all_known_within_selected8','common_known_class_images']:
      eligible=a['status']!='uncertain'and(support=='all_known_within_selected8'or(i,cat)in common)
      rr={k:r[k]for k in ['image_id','state_id','root_id','scene','split','camera']};rr.update(resolution=out['resolution'],seed=out['seed'],variant=variant,label_version=version,support=support,category=cat,status=a['status'],eligible=eligible,exclusion='none'if eligible else'label_uncertain'if a['status']=='uncertain'else'outside_common_known_support',GT_boxes=gt,predictions=pp,pred_count=len(pp))
      rr.update(match(pp,gt)if eligible else{'TP':None,'FP':None,'FN':None,'matches':[],'best_IoU_per_GT':[]});records.append(rr)
 groups=[]
 for res in[224,448]:
  for seed in[0,1]:
   for variant in ['original','corrected']:
    for version in labels:
     for support in ['all_known_within_selected8','common_known_class_images']:
      base=[x for x in records if(x['resolution'],x['seed'],x['variant'],x['label_version'],x['support'])==(res,seed,variant,version,support)]
      for split in ['selected8_all']+SPLITS:
       for cat in ['all_categories']+CATS:
        for camera in ['all',0,1]:
         rr=[x for x in base if(split=='selected8_all'or x['split']==split)and(cat=='all_categories'or x['category']==cat)and(camera=='all'or x['camera']==camera)]
         groups.append({'resolution':res,'seed':seed,'variant':variant,'label_version':version,'support':support,'split':split,'category':cat,'camera':camera,**aggregate(rr)})
 save(E/'results/sensitivity.json',{'utc':now(),'selected_ids':IDS,'records':records,'groups':groups,'prediction_source':str(source),'prediction_source_sha256':source_hash,'thresholds':{'score':.5,'saved_NMS_IoU':.3,'matching_IoU':.5},'decoder_rerun':False,'new_forward_passes':0,'new_training_steps':0,'new_features':0,'new_inputs':0,'full40_metrics_recomputed':False})
 csvsave(R/'all_group_metrics.csv',groups);csvsave(R/'all_image_metrics.csv',records)
 shutil.copy2(__file__,E/'source/score.py')
 save(E/'configs/sensitivity_frozen.json',{'utc':now(),'sha256':{str(p):sha(p)for p in [E/'results/sensitivity.json',E/'results/saved_predictions_selected8.json',E/'data/labels/denominator_changes.json',E/'source/score.py']}})
 print('EIGHT_IMAGE_SENSITIVITY_COMPLETE',len(records),'records',len(groups),'groups')
 for x in groups:
  if x['split']=='selected8_all'and x['camera']=='all'and x['category']=='all_categories':print(json.dumps({k:x[k]for k in ['resolution','seed','variant','label_version','support','scored_class_images','GT_boxes','TP','FP','FN']},ensure_ascii=False))
if __name__=='__main__':main()
