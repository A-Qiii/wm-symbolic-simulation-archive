from common import *
from net import iou
import csv
def one(pred,boxes):
 used=set();tp=[];fp=[]
 for j,p in enumerate(pred):
  choices=[(iou(p['box'],box),k)for k,box in enumerate(boxes)if k not in used];q=max(choices,default=(0,None))
  if q[0]>=.5:used.add(q[1]);tp.append({'prediction':j,'gt':q[1],'IoU':q[0]})
  else:fp.append(j)
 return {'TP':len(tp),'FP':len(fp),'FN':len(boxes)-len(used),'matches':tp,'false_positive_indices':fp,'false_negative_indices':[k for k in range(len(boxes))if k not in used]}
def aggregate(records):
 tp=sum(r['TP']for r in records);fp=sum(r['FP']for r in records);fn=sum(r['FN']for r in records);vals=[m['IoU']for r in records for m in r['matches']];return {'TP':tp,'FP':fp,'FN':fn,'precision':tp/(tp+fp)if tp+fp else None,'recall':tp/(tp+fn)if tp+fn else None,'F1':2*tp/(2*tp+fp+fn)if 2*tp+fp+fn else None,'matched_TP_mean_IoU':float(np.mean(vals))if vals else None,'known_image_classes':len(records)}
def main():
 assert not(E/'results/metrics.json').exists();frozen=read(E/'results/predictions_frozen.json');assert all(sha(p)==h for p,h in frozen['sha256'].items());rows=read(E/'data/public/images.json');ann=read(E/'data/annotation_private/labels.json');records=[];unknown=[]
 for seed in [0,1]:
  preds=read(E/f'data/predictions/decoded_seed{seed}.json');assert [r['image_id']for r in preds]==list(range(512))
  for r,a,pp in zip(rows,ann,preds):
   for c,cat in enumerate(CATS):
    ps=[p for p in pp['predictions']if p['category']==cat];base={k:r[k]for k in ['image_id','root_id','split','camera','sample']};base.update(seed=seed,category=cat)
    if not a['class_valid'][c]:unknown.append(dict(base,predictions_ignored=len(ps),reason=a['class_unknown_reasons'][c]));continue
    boxes=(np.array(a['boxes'][c])/np.array([r['width'],r['height'],r['width'],r['height']])).tolist();result=one(ps,boxes);records.append(dict(base,**result))
 metrics=[];roots=[]
 for seed in [0,1]:
  for split in ['train','dev','final']:
   for cam in ['all','static','gripper']:
    for cat in ['all']+CATS:
     rr=[r for r in records if r['seed']==seed and r['split']==split and(cam=='all'or r['camera']==cam)and(cat=='all'or r['category']==cat)];metrics.append({'seed':seed,'split':split,'camera':cam,'category':cat,**aggregate(rr)})
   for root in sorted({r['root_id']for r in rows if r['split']==split}):roots.append({'seed':seed,'split':split,'root_id':root,**aggregate([r for r in records if r['seed']==seed and r['root_id']==root])})
 save(E/'results/matches.json',records);save(E/'results/unknown_exclusions.json',unknown);save(E/'results/root_metrics.json',roots);save(E/'results/metrics.json',{'utc':now(),'evaluation_passes':1,'IoU_threshold':.5,'score_threshold':.5,'metrics':metrics,'root_dependence':'states within roots correlated; no iid frame confidence intervals','zero_certified_whole_image_negatives':True,'no_hyperparameter_or_checkpoint_selection':True})
 with(R/'metrics.csv').open('w')as f:w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
 for m in metrics:
  if m['camera']=='all':print(m,flush=True)
if __name__=='__main__':main()
