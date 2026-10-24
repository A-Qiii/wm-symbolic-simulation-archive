from common import *
from net import *
import collections,csv,shutil

def decode_without_peak(raw):
 with torch.inference_mode():
  logits,g=unpack(torch.tensor(raw));s=logits.sigmoid().numpy();boxes=toboxes(g,torch.arange(256)[:,None].expand(256,2)).clamp(0,1).numpy()
 out=[]
 for c in range(2):
  selected=[]
  for k in sorted(np.where(s[:,c]>=.5)[0],key=lambda k:(-s[k,c],int(k))):
   b=boxes[k,c]
   if b[2]<=b[0]or b[3]<=b[1]or any(iou(b,q['box'])>.3 for q in selected):continue
   selected.append({'category':CATS[c],'score':float(s[k,c]),'box':b.tolist(),'patch':int(k)})
   if len(selected)>=(1 if c==0 else 2):break
  out.extend(selected)
 return out

def counts(pred,gt):
 rem=set(range(len(gt)));tp=0
 for p in pred:
  k=max(rem,key=lambda j:iou(p['box'],gt[j]))if rem else None
  if k is not None and iou(p['box'],gt[k])>=.5:rem.remove(k);tp+=1
 return {'TP':tp,'FP':len(pred)-tp,'FN':len(gt)-tp}

def main():
 primary=read(E/'results/evaluation.json')['records'];diag=[];ablation=[]
 for file in sorted((E/'results/predictions').glob('*.npz')):
  seed=int(file.stem.rsplit('seed',1)[1]);z=np.load(file)
  for raw,i in zip(z['raw'],z['image_ids']):
   i=int(i);log,g=unpack(torch.tensor(raw));sc=log.sigmoid();peaks=(sc.T.reshape(1,2,16,16)==torch.nn.functional.max_pool2d(sc.T.reshape(1,2,16,16),3,1,1)).reshape(2,256).T;alt=decode_without_peak(raw)
   for j,c in enumerate(CATS):
    r=next(r for r in primary if r['seed']==seed and r['image_id']==i and r['category']==c)
    if r['status']=='uncertain':continue
    pp=[p for p in alt if p['category']==c];cc=counts(pp,r['GT_boxes']);ablation.append({k:r[k]for k in ['seed','image_id','split','category']}|{'primary_TP':r['TP'],'primary_FP':r['FP'],'primary_FN':r['FN'],'no_peak_TP':cc['TP'],'no_peak_FP':cc['FP'],'no_peak_FN':cc['FN']})
    for gi,b in enumerate(r['GT_boxes']):
     center=(np.array(b[:2])+b[2:])/2;cell=np.minimum((center*16).astype(int),15);p=int(cell[1]*16+cell[0]);pb=toboxes(g[p,j],torch.tensor(p)).clamp(0,1).numpy();score=float(sc[p,j]);iv=iou(pb,b);matched=any(m['GT']==gi for m in r['matches']);diag.append({k:r[k]for k in ['seed','image_id','split','camera','category']}|{'GT_index':gi,'target_center_patch':p,'target_center_score':score,'target_center_predicted_box_IoU':iv,'target_center_is_local_max':bool(peaks[p,j]),'primary_matched':matched,'failure_evidence':'matched'if matched else'center_score_below_threshold'if score<.5 else'center_suppressed_by_3x3_peak'if not peaks[p,j]else'center_box_IoU_below_0.5'if iv<.5 else'NMS_or_cap_or_other_prediction_matching','limit':'GT-center audit uses annotation only for posthoc analysis; another patch could localize object; not an alternate inference oracle'})
 by=collections.defaultdict(list)
 for r in ablation:by[r['seed'],r['split'],r['category']].append(r)
 grouped=[]
 for (seed,split,c),rr in by.items():grouped.append({'seed':seed,'split':split,'category':c,**{k:sum(r[k]for r in rr)for k in ['primary_TP','primary_FP','primary_FN','no_peak_TP','no_peak_FP','no_peak_FN']}})
 save(E/'results/failure_audit.json',{'utc':now(),'records':diag,'one_component_diagnostic':'remove3x3localpeak on saved outputs only; same.5threshold,.3NMS,samecaps; no new forward/fitting/model adoption','ablation_by_image':ablation,'ablation_groups':grouped})
 with(R/'failure_by_GT.csv').open('w')as f:wr=csv.DictWriter(f,fieldnames=list(diag[0]));wr.writeheader();wr.writerows(diag)
 with(R/'decode_diagnostic.csv').open('w')as f:wr=csv.DictWriter(f,fieldnames=list(grouped[0]));wr.writeheader();wr.writerows(grouped)
 for p in Path(__file__).parent.glob('diagnose.py'):shutil.copy2(p,E/'source'/p.name)
 print('READONLY_LOCALIZER_FAILURE_AUDIT_COMPLETE',len(diag),flush=True)
 print(json.dumps(grouped,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
