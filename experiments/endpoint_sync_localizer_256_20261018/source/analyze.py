from common import *
from net import unpack,toboxes
import torch
def main():
 assert not(E/'results/confidence_geometry_audit.json').exists();rows=read(E/'data/public/images.json');anns=read(E/'data/annotation_private/labels.json');matches=read(E/'results/matches.json');summary=[]
 for seed in [0,1]:
  for split in ['train','dev','final']:
   z=np.load(E/f'data/predictions/raw_seed{seed}_{split}.npz');byid={int(i):a for i,a in zip(z['image_ids'],z['raw'])};decoded=read(E/f'data/predictions/decoded_seed{seed}.json')
   for cat in CATS:
    c=CATS.index(cat);slots=[0]if c==0 else[1,2];rr=[r for r in matches if r['seed']==seed and r['split']==split and r['category']==cat];counts={'FN_with_no_raw_box_IoU_at_least_05':0,'FN_with_adequate_geometry_but_all_scores_below_05':0,'FN_with_adequate_above_threshold_raw_candidate':0,'FP_center_on_known_handle':0};center_scores=[];center_ious=[]
    for rec in rr:
     i=rec['image_id'];row=rows[i];a=anns[i];raw=byid[i]
     with torch.inference_mode():log,g=unpack(torch.tensor(raw));bb=toboxes(g,torch.arange(1024)[:,None].expand(-1,3)).clamp(0,1).numpy();scores=log.sigmoid().numpy()
     gt=np.array(a['boxes'][c])/[row['width'],row['height'],row['width'],row['height']];candidates=bb[:,slots].reshape(-1,4);ss=scores[:,slots].reshape(-1)
     for j,b in enumerate(gt):
      inter=np.maximum(np.minimum(candidates[:,2:],b[2:])-np.maximum(candidates[:,:2],b[:2]),0).prod(1);union=np.maximum(candidates[:,2:]-candidates[:,:2],0).prod(1)+np.prod(b[2:]-b[:2])-inter;ov=inter/np.maximum(union,1e-9);center=(b[:2]+b[2:])/2;xy=np.minimum((center*32).astype(int),31);p=int(xy[1]*32+xy[0]);center_scores.append(float(scores[p,slots].max()));center_ious.append(float(ov.reshape(1024,len(slots))[p].max()))
      if j not in rec['false_negative_indices']:continue
      good=ov>=.5
      if not good.any():counts['FN_with_no_raw_box_IoU_at_least_05']+=1
      elif float(ss[good].max())<.5:counts['FN_with_adequate_geometry_but_all_scores_below_05']+=1
      else:counts['FN_with_adequate_above_threshold_raw_candidate']+=1
     if cat=='gripper_finger'and rec['FP']:
      sem=np.load(a['semantic_file'])['drawer_parts'];ps=[p for p in decoded[i]['predictions']if p['category']==cat]
      for j in rec['false_positive_indices']:
       b=ps[j]['box'];xy=(np.array(b[:2])+b[2:])/2*[row['width'],row['height']];x=min(int(xy[0]),row['width']-1);y=min(int(xy[1]),row['height']-1);counts['FP_center_on_known_handle']+=int(sem[y,x]==2)
    summary.append({'seed':seed,'split':split,'category':cat,'GT_boxes':len(center_scores),'FN':sum(r['FN']for r in rr),'FP':sum(r['FP']for r in rr),**counts,'GT_center_patch_max_slot_score_median':float(np.median(center_scores)),'GT_center_patch_max_slot_IoU_median':float(np.median(center_ious))})
 save(E/'results/confidence_geometry_audit.json',{'utc':now(),'observational_only_no_parameter_or_threshold_search':True,'caveat':'Raw candidate existence is oracle-assisted analysis, not detector recall. Matching/NMS overlap can explain remaining above-threshold FN. Handle-center test is narrow and may miss other forms of confusion.','rows':summary})
if __name__=='__main__':main()
