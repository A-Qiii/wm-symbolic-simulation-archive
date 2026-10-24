from common import *
from net import *
import collections,csv,time

def aggregate(rr):
 known=[r for r in rr if r['status']!='uncertain'];unknown=[r for r in rr if r['status']=='uncertain'];tp=sum(r['TP']for r in known);fp=sum(r['FP']for r in known);fn=sum(r['FN']for r in known);ious=[z for r in known for z in r['best_IoU_per_GT']];absent=[r for r in known if r['status']=='not_visible']
 return {'class_images':len(rr),'known_class_images':len(known),'unknown_class_images':len(unknown),'GT_boxes':tp+fn,'TP':tp,'FP':fp,'FN':fn,'precision':tp/(tp+fp)if tp+fp else None,'recall':tp/(tp+fn)if tp+fn else None,'mean_best_IoU_including_misses':float(np.mean(ious))if ious else None,'unknown_images_with_prediction':sum(r['pred_count']>0 for r in unknown),'known_not_visible_images':len(absent),'not_visible_FP_image_rate':sum(r['pred_count']>0 for r in absent)/len(absent)if absent else None}
def csvsave(path,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with Path(path).open('w')as f:
  wr=csv.DictWriter(f,fieldnames=keys);wr.writeheader();wr.writerows(rows)
def main():
 assert not(E/'results/evaluation.json').exists();check_manifest(E/'configs/final_heads_frozen.json')
 rows=read(E/'data/public/images.json');anns=read(E/'data/annotations/final.json')['annotations'];torch.set_num_threads(4);record=[];images=[];boxaudit=[];elapsed=[];(E/'results/predictions').mkdir()
 for res in(224,448):
  grid=res//14;n=grid*grid
  for seed in(0,1):
   q=make().eval().requires_grad_(False);q.load_state_dict(torch.load(E/f'models/r{res}/localizer_seed{seed}_update0600.pt',weights_only=True,map_location='cpu')['state']);t0=time.perf_counter()
   for split in SPLITS:
    path=E/('data/train'if split=='train'else'data/evaluation')/f'r{res}_{split}.npz';d=np.load(path)
    with torch.inference_mode():raw=q(torch.tensor(d['x'])).numpy()
    if res==224:
     ref=np.load(BASE/f'results/predictions/{split}_seed{seed}.npz');assert np.array_equal(raw,ref['raw'])and np.array_equal(d['image_ids'],ref['image_ids']),('224 prediction changed',seed,split)
    np.savez_compressed(E/f'results/predictions/r{res}_{split}_seed{seed}.npz',raw=raw,image_ids=d['image_ids'])
    for k,i in enumerate(d['image_ids']):
     i=int(i);r=rows[i];preds=decode(raw[k],grid);images.append({'resolution':res,'seed':seed,'image_id':i,'predictions':preds})
     logits,g=unpack(torch.tensor(raw[k]));scores=logits.sigmoid();peaks=(scores.T.reshape(1,2,grid,grid)==F.max_pool2d(scores.T.reshape(1,2,grid,grid),3,1,1)).reshape(2,n).T
     for ci,c in enumerate(CATS):
      ann=next(a for a in anns[i]['classes']if a['category']==c);pp=[p for p in preds if p['category']==c];gt=[(np.array(b)/[r['width'],r['height'],r['width'],r['height']]).tolist()for b in ann['boxes']];rr={key:r[key]for key in ['image_id','state_id','root_id','scene','split','camera']};rr.update(resolution=res,seed=seed,category=c,status=ann['status'],reason=ann['reason'],GT_count=len(gt),pred_count=len(pp),GT_boxes=gt,predictions=pp)
      if ann['status']=='uncertain':rr.update(TP=None,FP=None,FN=None,best_IoU_per_GT=[],matches=[])
      else:
       remaining=set(range(len(gt)));matches=[]
       for pi,p in enumerate(pp):
        gi=max(remaining,key=lambda z:iou(p['box'],gt[z]))if remaining else None;iv=iou(p['box'],gt[gi])if gi is not None else 0
        if gi is not None and iv>=.5:remaining.remove(gi);matches.append({'prediction':pi,'GT':gi,'IoU':iv})
       tp=len(matches);rr.update(TP=tp,FP=len(pp)-tp,FN=len(gt)-tp,best_IoU_per_GT=[max([iou(p['box'],gg)for p in pp],default=0)for gg in gt],matches=matches)
       for j,gg in enumerate(gt):
        z=np.array(gg);center=(z[:2]+z[2:])/2;cell=np.minimum((center*grid).astype(int),grid-1);p=int(cell[1]*grid+cell[0]);rawbox=toboxes(g[p,ci],torch.tensor(p),grid).clamp(0,1).numpy();score=float(scores[p,ci]);overlap=iou(rawbox,z);matched=any(m['GT']==j for m in matches)
        boxaudit.append({'resolution':res,'seed':seed,'image_id':i,'root_id':r['root_id'],'split':split,'scene':r['scene'],'camera':r['camera'],'category':c,'GT_index':j,'short_side_patch224':float(min(z[2:]-z[:2])*16),'width_RGB':ann['boxes'][j][2]-ann['boxes'][j][0],'height_RGB':ann['boxes'][j][3]-ann['boxes'][j][1],'matched':matched,'best_IoU':rr['best_IoU_per_GT'][j],'GT_center_score':score,'GT_center_box_IoU':overlap,'GT_center_local_peak':bool(peaks[p,ci]),'GT_center_patch':p,'first_center_gate_failure':('matched'if matched else'center_score_below_.5'if score<.5 else'center_box_IoU_below_.5'if overlap<.5 else'center_not_peak'if not peaks[p,ci]else'other_decoder_or_competition'),'note':'GT center only for audit, never fed into inference'})
      record.append(rr)
   elapsed.append({'resolution':res,'seed':seed,'CPU_full_eval_seconds_including_IO_and_audit':time.perf_counter()-t0})
 groups=[]
 for res in(224,448):
  for seed in(0,1):
   subset=[r for r in record if r['resolution']==res and r['seed']==seed]
   for dims in [('split','category'),('split','scene','camera','category'),('split','camera','category'),('split','category','status'),('split','category','reason')]:
    by=collections.defaultdict(list)
    for r in subset:by[tuple(r[d]for d in dims)].append(r)
    for key,rr in by.items():groups.append({'resolution':res,'seed':seed,'dimensions':'/'.join(dims),**dict(zip(dims,key)),**aggregate(rr)})
 old_eval=read(BASE/'results/evaluation.json');old_records={(r['seed'],r['image_id'],r['category']):r for r in old_eval['records']}
 for r in record:
  if r['resolution']==224:
   old=old_records[r['seed'],r['image_id'],r['category']]
   for k in ['TP','FP','FN','GT_boxes','predictions','matches','best_IoU_per_GT']:assert r[k]==old[k],('224 evaluation differs',k,r['image_id'])
 save(E/'results/evaluation.json',{'utc':now(),'threshold':.5,'IoU_threshold':.5,'NMS_IoU':.3,'no_tuning':True,'224_predictions_and_metrics_exactly_reproduced':True,'records':record,'groups':groups,'timing':elapsed})
 save(E/'results/decoded_predictions.json',images);save(E/'results/GT_center_audit.json',boxaudit)
 csvsave(R/'localization_by_group.csv',groups);csvsave(R/'localization_by_image.csv',record);csvsave(R/'GT_center_audit.csv',boxaudit)
 print('RESOLUTION_EVALUATION_COMPLETE',len(record),'class-image records',len(boxaudit),'GT records',flush=True)
 for r in groups:
  if r['dimensions']=='split/category':print(json.dumps(r,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
