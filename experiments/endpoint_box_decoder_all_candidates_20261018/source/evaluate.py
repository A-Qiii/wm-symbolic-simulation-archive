from common import *
import collections,csv,shutil

def aggregate(rr):
 known=[x for x in rr if x['status']!='uncertain'];unknown=[x for x in rr if x['status']=='uncertain'];tp=sum(x['TP']for x in known);fp=sum(x['FP']for x in known);fn=sum(x['FN']for x in known);ious=[z for x in known for z in x['best_IoU_per_GT']]
 return {'class_images':len(rr),'known_class_images':len(known),'unknown_class_images':len(unknown),'GT_boxes':tp+fn,'TP':tp,'FP':fp,'FN':fn,'precision':tp/(tp+fp)if tp+fp else None,'recall':tp/(tp+fn)if tp+fn else None,'mean_best_IoU_including_misses':float(np.mean(ious))if ious else None,'unknown_images_with_prediction':sum(x['pred_count']>0 for x in unknown),'known_not_visible_images':sum(x['status']=='not_visible'for x in known),'not_visible_FP_rate':None,'unmatched_predictions_overlapping_GT_at_least_.5':sum(x['unmatched_overlapping_GT']for x in known)}
def csvsave(path,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with Path(path).open('w')as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def main():
 assert not(E/'results/evaluation.json').exists();check_manifest(E/'configs/outputs_frozen_before_scoring.json')
 rows=read(E/'data/images.json');anns=read(E/'data/annotations/final.json')['annotations'];outputs=read(E/'results/decoded_predictions.json');traces=read(E/'results/candidate_trace.json');by=collections.defaultdict(list)
 for x in traces:by[x['resolution'],x['seed'],x['image_id'],x['category']].append(x)
 records=[];audit=[];original=read(BASE/'results/evaluation.json');old_index={(x['resolution'],x['seed'],x['image_id'],x['category']):x for x in original['records']}
 for item in outputs:
  res,seed,i=item['resolution'],item['seed'],item['image_id'];r=rows[i]
  for ci,c in enumerate(CATS):
   a=anns[i]['classes'][ci];assert a['category']==c;gt=[(np.array(b)/[r['width'],r['height'],r['width'],r['height']]).tolist()for b in a['boxes']];pair={}
   for variant in ['original','corrected']:
    pp=[p for p in item[variant]if p['category']==c];rr={k:r[k]for k in ['image_id','state_id','root_id','scene','split','camera']};rr.update(resolution=res,seed=seed,variant=variant,category=c,status=a['status'],reason=a['reason'],GT_boxes=gt,GT_count=len(gt),predictions=pp,pred_count=len(pp))
    if a['status']=='uncertain':rr.update(TP=None,FP=None,FN=None,matches=[],best_IoU_per_GT=[],unmatched_overlapping_GT=None,FP_details=[])
    else:
     rem=set(range(len(gt)));matches=[]
     for pi,p in enumerate(pp):
      gi=max(rem,key=lambda z:iou(p['box'],gt[z]))if rem else None;iv=iou(p['box'],gt[gi])if gi is not None else 0
      if gi is not None and iv>=.5:rem.remove(gi);matches.append({'prediction':pi,'GT':gi,'IoU':iv})
     matched_p={x['prediction']for x in matches};fps=[{'prediction':k,'patch':p['patch'],'score':p['score'],'best_GT_IoU':max((iou(p['box'],g)for g in gt),default=0)}for k,p in enumerate(pp)if k not in matched_p]
     rr.update(TP=len(matches),FP=len(pp)-len(matches),FN=len(gt)-len(matches),matches=matches,best_IoU_per_GT=[max((iou(p['box'],g)for p in pp),default=0)for g in gt],unmatched_overlapping_GT=sum(x['best_GT_IoU']>=.5 for x in fps),FP_details=fps)
    if variant=='original':
     old=old_index[res,seed,i,c]
     for k in ['TP','FP','FN','predictions','matches','best_IoU_per_GT']:assert rr[k]==old[k],('old metrics mismatch',res,seed,i,c,k)
    records.append(rr);pair[variant]=rr
   if a['status']=='uncertain':continue
   cc=by[res,seed,i,c]
   for j,g in enumerate(gt):
    qualifying=[dict(patch=x['patch'],score=x['score'],IoU=iou(x['box'],g),original=x['original_disposition'],corrected=x['corrected_disposition'])for x in cc if x['valid_box']and iou(x['box'],g)>=.5]
    om=any(x['GT']==j for x in pair['original']['matches']);nm=any(x['GT']==j for x in pair['corrected']['matches']);kept=[x for x in qualifying if x['corrected']['reason']=='kept']
    cause='matched'if nm else'no_qualified_candidate'if not qualifying else'NMS_removed_all_qualified'if not kept else'one_to_one_matching_competition'
    center=(np.array(g[:2])+g[2:])/2;cell=np.minimum((center*(res//14)).astype(int),res//14-1);cp=int(cell[1]*(res//14)+cell[0]);center_entry=next((x for x in qualifying if x['patch']==cp),None)
    audit.append({k:r[k]for k in ['image_id','root_id','scene','split','camera']}|{'resolution':res,'seed':seed,'category':c,'GT_index':j,'original_matched':om,'corrected_matched':nm,'qualified_candidate_count':len(qualifying),'corrected_miss_cause':cause,'GT_center_patch':cp,'GT_center_qualified_candidate':center_entry,'qualified_candidates':qualifying,'original_best_IoU':pair['original']['best_IoU_per_GT'][j],'corrected_best_IoU':pair['corrected']['best_IoU_per_GT'][j],'limit':'GT used only after decoder freeze for explanation; candidate existence is not a scored detector result'})
 groups=[]
 for res in[224,448]:
  for seed in[0,1]:
   for variant in ['original','corrected']:
    subset=[r for r in records if r['resolution']==res and r['seed']==seed and r['variant']==variant]
    for dims in [('split','category'),('split','camera','category'),('split','scene','camera','category'),('split','category','status')]:
     bydim=collections.defaultdict(list)
     for rr in subset:bydim[tuple(rr[d]for d in dims)].append(rr)
     for key,rr in bydim.items():groups.append({'resolution':res,'seed':seed,'variant':variant,'dimensions':'/'.join(dims),**dict(zip(dims,key)),**aggregate(rr)})
 causes=[]
 for res in[224,448]:
  for seed in[0,1]:
   for split in SPLITS:
    for c in CATS:
     z=[x for x in audit if x['resolution']==res and x['seed']==seed and x['split']==split and x['category']==c];causes.append({'resolution':res,'seed':seed,'split':split,'category':c,'GT_boxes':len(z),'original_TP':sum(x['original_matched']for x in z),'corrected_TP':sum(x['corrected_matched']for x in z),'gained_GT':sum(not x['original_matched']and x['corrected_matched']for x in z),'lost_GT':sum(x['original_matched']and not x['corrected_matched']for x in z),'corrected_miss_causes':dict(collections.Counter(x['corrected_miss_cause']for x in z if not x['corrected_matched']))})
 save(E/'results/evaluation.json',{'utc':now(),'records':records,'groups':groups,'score_threshold':.5,'NMS_IoU':.3,'matching_IoU':.5,'no_GT_candidate_filtering':True,'original_metrics_exact':True});save(E/'results/GT_candidate_audit.json',audit);save(E/'results/candidate_cause_summary.json',causes)
 csvsave(R/'all_group_metrics.csv',groups);csvsave(R/'all_image_metrics.csv',records);csvsave(R/'GT_candidate_audit.csv',audit);csvsave(R/'candidate_cause_summary.csv',causes)
 shutil.copy2(__file__,E/'source/evaluate.py')
 print('ALL_CANDIDATE_EVALUATION_COMPLETE',len(records),len(audit),flush=True)
 for x in groups:
  if x['dimensions']=='split/category'and x['variant']=='corrected':print(json.dumps(x,ensure_ascii=False),flush=True)
 print('MAIN_448_AB_MISS_CAUSES',json.dumps([x for x in causes if x['resolution']==448 and x['split']==SPLITS[1]],ensure_ascii=False),flush=True)
if __name__=='__main__':main()
