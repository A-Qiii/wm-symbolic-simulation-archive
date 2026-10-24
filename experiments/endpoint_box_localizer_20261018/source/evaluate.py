from common import *
from net import *
import collections,csv

def aggregate(rr):
 known=[r for r in rr if r['status']!='uncertain'];unknown=[r for r in rr if r['status']=='uncertain'];tp=sum(r['TP']for r in known);fp=sum(r['FP']for r in known);fn=sum(r['FN']for r in known);positives=[r for r in known if r['GT_count']];absent=[r for r in known if r['status']=='not_visible'];ious=[z for r in known for z in r['best_IoU_per_GT']]
 return {'class_images':len(rr),'known_class_images':len(known),'unknown_class_images':len(unknown),'GT_boxes':tp+fn,'TP':tp,'FP':fp,'FN':fn,'precision':tp/(tp+fp)if tp+fp else None,'recall':tp/(tp+fn)if tp+fn else None,'mean_best_IoU_including_misses':float(np.mean(ious))if ious else None,'median_best_IoU_including_misses':float(np.median(ious))if ious else None,'positive_class_images':len(positives),'positive_images_with_any_prediction':sum(r['pred_count']>0 for r in positives),'unknown_images_with_prediction':sum(r['pred_count']>0 for r in unknown),'known_not_visible_images':len(absent),'not_visible_images_with_prediction':sum(r['pred_count']>0 for r in absent),'not_visible_FP_image_rate':sum(r['pred_count']>0 for r in absent)/len(absent)if absent else None}

def main():
 assert not(E/'results/evaluation.json').exists()
 for p,h in read(E/'configs/final_heads_frozen.json')['sha256'].items():assert sha(p)==h,p
 rows=read(E/'data/public/images.json');anns=read(E/'data/annotations/final.json')['annotations'];torch.set_num_threads(4);record=[];images=[];(E/'results/predictions').mkdir()
 for seed in(0,1):
  q=make().eval().requires_grad_(False);q.load_state_dict(torch.load(E/f'models/localizer_seed{seed}_update0600.pt',weights_only=True,map_location='cpu')['state'])
  for path in [E/'data/train/train.npz']+sorted((E/'data/evaluation').glob('*.npz')):
   d=np.load(path)
   with torch.inference_mode():raw=q(torch.tensor(d['x'])).numpy()
   np.savez_compressed(E/f'results/predictions/{path.stem}_seed{seed}.npz',raw=raw,image_ids=d['image_ids'])
   for k,i in enumerate(d['image_ids']):
    i=int(i);r=rows[i];preds=decode(raw[k]);images.append({'seed':seed,'image_id':i,'predictions':preds})
    for c in CATS:
     ann=next(a for a in anns[i]['classes']if a['category']==c);pp=[p for p in preds if p['category']==c];gt=[(np.array(b)/[r['width'],r['height'],r['width'],r['height']]).tolist()for b in ann['boxes']];rr={key:r[key]for key in ['image_id','state_id','root_id','scene','split','camera']};rr.update(seed=seed,category=c,status=ann['status'],reason=ann['reason'],GT_count=len(gt),pred_count=len(pp),GT_boxes=gt,predictions=pp)
     if ann['status']=='uncertain':rr.update(TP=None,FP=None,FN=None,best_IoU_per_GT=[],matches=[])
     else:
      remaining=set(range(len(gt)));matches=[]
      for pi,p in enumerate(pp):
       g=max(remaining,key=lambda z:iou(p['box'],gt[z]))if remaining else None;iv=iou(p['box'],gt[g])if g is not None else 0
       if g is not None and iv>=.5:remaining.remove(g);matches.append({'prediction':pi,'GT':g,'IoU':iv})
      tp=len(matches);rr.update(TP=tp,FP=len(pp)-tp,FN=len(gt)-tp,best_IoU_per_GT=[max([iou(p['box'],g)for p in pp],default=0)for g in gt],matches=matches)
     record.append(rr)
 groups=[]
 for seed in(0,1):
  subset=[r for r in record if r['seed']==seed]
  for dims in [('split','category'),('split','scene','camera','category'),('split','camera','category'),('split','category','status'),('split','category','reason')]:
   by=collections.defaultdict(list)
   for r in subset:by[tuple(r[d]for d in dims)].append(r)
   for key,rr in by.items():groups.append({'seed':seed,'dimensions':'/'.join(dims),**dict(zip(dims,key)),**aggregate(rr)})
 save(E/'results/evaluation.json',{'utc':now(),'threshold':.5,'IoU_threshold':.5,'NMS_IoU':.3,'no_tuning':True,'records':record,'groups':groups});save(E/'results/decoded_predictions.json',images)
 with(R/'localization_by_image.csv').open('w')as f:
  keys=['seed','image_id','root_id','scene','split','camera','category','status','reason','GT_count','pred_count','TP','FP','FN','best_IoU_per_GT'];wr=csv.DictWriter(f,fieldnames=keys);wr.writeheader();wr.writerows({k:r[k]for k in keys}for r in record)
 with(R/'localization_by_group.csv').open('w')as f:
  keys=list(dict.fromkeys(k for r in groups for k in r));wr=csv.DictWriter(f,fieldnames=keys);wr.writeheader();wr.writerows(groups)
 print('LOCALIZATION_EVALUATION_COMPLETE',len(record),'class-image records',flush=True)
 for r in groups:
  if r['dimensions']=='split/category':print(json.dumps(r,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
