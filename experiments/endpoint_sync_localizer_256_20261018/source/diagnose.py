from common import *
from net import *
from score import one
def main():
 assert not(E/'results/single_counterfactual.json').exists();rows=read(E/'data/public/images.json');ann=read(E/'data/annotation_private/labels.json');matches=read(E/'results/matches.json');pred=read(E/'data/predictions/decoded_seed0.json');errors=[r for r in matches if r['seed']==0 and r['split']=='final'and(r['FN']or r['FP'])]
 if not errors:save(E/'results/single_counterfactual.json',{'no_errors':True});return
 # First error in fixed image/class order, not chosen for a successful intervention.
 q=errors[0];i=q['image_id'];r=rows[i];c=CATS.index(q['category']);gt=np.array(ann[i]['boxes'][c])/[r['width'],r['height'],r['width'],r['height']];rawfile=np.load(E/'data/predictions/raw_seed0_final.npz');raw=rawfile['raw'][np.where(rawfile['image_ids']==i)[0][0]].copy();shaped=raw.reshape(1024,3,5);slots=[0]if c==0 else[1,2];before=[p for p in pred[i]['predictions']if p['category']==CATS[c]]
 if q['FN']:
  target=gt[q['false_negative_indices'][0]]
  with torch.inference_mode():log,g=unpack(torch.tensor(raw));boxes=toboxes(g,torch.arange(1024)[:,None].expand(-1,3)).clamp(0,1).numpy();scores=log.sigmoid().numpy()
  viable=[(float(scores[p,s]),p,s,iou(boxes[p,s],target))for p in range(1024)for s in slots if iou(boxes[p,s],target)>=.5];viable.sort(reverse=True)
  if viable:
   score,p,s,overlap=viable[0];old=float(shaped[p,s,0]);shaped[p,s,0]=20.;intervention={'component':'one candidate confidence logit only','oracle_value':20.,'patch':p,'slot':s,'previous_score':score,'previous_logit':old,'existing_box_IoU_to_missed_GT':overlap,'interpretation':'candidate geometry already meets IoU .5; tests score gating / score-order NMS route without changing geometry or global threshold'}
  else:
   center=(target[:2]+target[2:])/2;cell=np.minimum((center*32).astype(int),31);p=int(cell[1]*32+cell[0]);s=max(slots,key=lambda t:float(scores[p,t]));offset=center*32-(cell+.5);wh=target[2:]-target[:2];shaped[p,s,1:3]=np.arctanh(np.clip(offset/.5,-.999999,.999999));shaped[p,s,3:5]=np.log(wh/(1-wh));intervention={'component':'one candidate geometry only','patch':p,'slot':s,'unchanged_score':float(scores[p,s]),'oracle_box':target.tolist(),'interpretation':'no existing candidate reaches IoU .5; tests GT geometry replacement at GT center patch, keeping confidence and global decoder fixed'}
 else:
  fp=before[q['false_positive_indices'][0]];p,s=fp['patch'],fp['slot'];shaped[p,s,0]=-20.;intervention={'component':'one false-positive candidate confidence only','patch':p,'slot':s,'previous_score':fp['score'],'oracle_value':-20.,'interpretation':'tests this single extra candidate; other candidates may be exposed by NMS'}
 after=[p for p in decode(raw)if p['category']==CATS[c]];save(E/'results/single_counterfactual.json',{'utc':now(),'seed':0,'image_id':i,'root_id':r['root_id'],'sample':r['sample'],'camera':r['camera'],'category':CATS[c],'selection':'first final image/class with any error, fixed order','intervention':intervention,'before':one(before,gt),'counterfactual':one(after,gt),'number_of_interventions':1,'main_metrics_unchanged':True,'weights_unchanged':True,'scope':'Oracle-assisted failure diagnostic only; never RGB-only localization score or deployed input','predictions_before':before,'predictions_counterfactual':after})
if __name__=='__main__':main()
