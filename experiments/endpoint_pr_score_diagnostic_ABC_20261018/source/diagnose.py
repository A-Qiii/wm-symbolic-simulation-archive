from common import *
import sys,os,time,csv,torch
def unpack(raw):
 z=torch.as_tensor(raw).reshape(1024,3,5);scores=z[...,0].sigmoid();g=torch.cat([.5*z[...,1:3].tanh(),z[...,3:5].sigmoid()],-1);return scores.numpy(),toboxes(g).clamp(0,1).numpy()
def toboxes(g):
 p=torch.arange(1024)[:,None].expand(-1,3);xy=torch.stack([p%32+.5,p//32+.5],-1).to(g.dtype);center=(xy+g[...,:2])/32;return torch.cat([center-g[...,2:]/2,center+g[...,2:]/2],-1)
def overlaps(box,boxes):
 inter=np.maximum(np.minimum(box[2:],boxes[:,2:])-np.maximum(box[:2],boxes[:,:2]),0).prod(1);union=np.maximum(box[2:]-box[:2],0).prod()+np.maximum(boxes[:,2:]-boxes[:,:2],0).prod(1)-inter;return inter/np.maximum(union,1e-9)
def nms(boxes,order):
 keep=[];remaining=order.copy()
 while len(remaining):
  i=int(remaining[0]);keep.append(i);rest=remaining[1:]
  if not len(rest):break
  remaining=rest[overlaps(boxes[i],boxes[rest])<=.5]
 return np.array(keep,np.int32)
def reference_nms(boxes,order):
 def scalar_iou(a,b):
  inter=np.maximum(np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]),0).prod();u=np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-inter;return float(inter/max(float(u),1e-9))
 keep=[]
 for i in order:
  if not any(scalar_iou(boxes[i],boxes[j])>.5 for j in keep):keep.append(int(i))
 return np.array(keep,np.int32)
def match(boxes,gt):
 used=set();tp=np.zeros(len(boxes),np.uint8);matched=np.full(len(boxes),-1,np.int32);miou=np.zeros(len(boxes),np.float32)
 for j,box in enumerate(boxes):
  ovs=overlaps(box,gt);best=max(((float(v),k)for k,v in enumerate(ovs)if k not in used),default=(0.,-1))
  if best[0]>=.5:used.add(best[1]);tp[j]=1;matched[j]=best[1];miou[j]=best[0]
 return tp,matched,miou
def point(scores,tp,n_gt,threshold):
 selected=scores>=threshold;nt=int(tp[selected].sum());nf=int(selected.sum())-nt;return {'threshold':float(threshold),'TP':nt,'FP':nf,'FN':n_gt-nt,'precision':nt/(nt+nf)if nt+nf else None,'recall':nt/n_gt,'F1':2*nt/(n_gt+nt+nf)}
def pr(scores,tp,image,patch,slot,n_gt):
 order=np.lexsort((slot,patch,image,-scores));s=scores[order];t=tp[order];ct=np.cumsum(t,dtype=np.int64);cf=np.arange(1,len(t)+1)-ct;ends=np.r_[np.where(s[:-1]!=s[1:])[0],len(s)-1];threshold=s[ends];precision=ct[ends]/(ct[ends]+cf[ends]);recall=ct[ends]/n_gt;mrec=np.r_[0.,recall,1.];mpre=np.r_[0.,precision,0.];mpre=np.maximum.accumulate(mpre[::-1])[::-1];change=np.where(mrec[1:]!=mrec[:-1])[0];ap=float(np.sum((mrec[change+1]-mrec[change])*mpre[change+1]));return {'threshold':threshold,'precision':precision,'recall':recall,'TP':ct[ends],'FP':cf[ends],'precision_envelope':mpre[1:-1]},ap
def main():
 assert not(E/'results/summary.json').exists();protocol=read(E/'configs/protocol.json');allowed=set(protocol['allowed_input_files']);opened=set()
 # Analytic metric guards: missed GT stays in denominator and tied scores form one threshold point.
 _,ap=pr(np.array([.9,.8,.7]),np.array([1,0,1]),np.zeros(3),np.arange(3),np.zeros(3),2);assert abs(ap-5/6)<1e-12
 _,ap=pr(np.array([.9,.8]),np.array([1,0]),np.zeros(2),np.arange(2),np.zeros(2),2);assert abs(ap-.5)<1e-12
 _,ap=pr(np.array([.9,.9]),np.array([1,0]),np.zeros(2),np.arange(2),np.zeros(2),1);assert abs(ap-.5)<1e-12
 def guard(event,args):
  if event=='open'and isinstance(args[0],(str,bytes)):
   path=str(Path(os.fsdecode(args[0])).resolve());p=Path(path)
   if p.is_relative_to(BASE):
    mode=args[1];flags=args[2]if len(args)>2 else 0
    if (isinstance(mode,str)and any(c in mode for c in 'wa+'))or(isinstance(flags,int)and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)):raise RuntimeError('Old experiment write prohibited: '+path)
    if path not in allowed:raise RuntimeError('Input is not AB/C allowlisted: '+path)
    opened.add(path)
 sys.addaudithook(guard);torch.set_num_threads(4);start=time.perf_counter();assert all(sha(p)==h for p,h in read(E/'configs/input_frozen.json')['sha256'].items());rows=[]
 for scene in 'ABC':
  for root in range(8):
   for sample in range(8):
    for cam in ['static','gripper']:
     d=BASE/'data/annotation_private'/f'sync256_{scene}_{root:02d}'/f's{sample:02d}';a=read(d/f'labels_{cam}.json');meta=read(d/f'camera_{cam}.json');rows.append({'image_id':a['image_id'],'root_id':a['root_id'],'sample':sample,'camera':cam,'scene':scene,'split':'train'if scene in 'AB'else'dev','width':meta['width'],'height':meta['height'],'class_valid':a['class_valid'],'boxes':a['boxes'],'unknown_reasons':a['class_unknown_reasons']})
 assert [r['image_id']for r in rows]==list(range(384));save(E/'data/ABC_labels_metadata.json',rows);slot_audit={};conflicts=[]
 z=np.load(BASE/'data/train/train.npz',allow_pickle=False);y=z['y'];target=z['b'];valid=z['v'];ids=z['image_ids'];assert np.array_equal(ids,np.arange(256));ii,pp,ss=np.where(y>0)
 for s in range(3):
  images=np.unique(ii[ss==s]);slot_audit[str(s)]={'positive_centers':int((ss==s).sum()),'positive_images':len(images),'positive_roots':len({rows[i]['root_id']for i in images}),'valid_image_patch_positions':int(valid[:,s].sum())*1024,'positive_image_ids':images.tolist()}
 for r in rows:
  if r['class_valid'][1]:
   boxes=np.array(r['boxes'][1]);assert len(boxes)==2;ov=float(overlaps(boxes[0].astype(float),boxes[1:].astype(float))[0])
   if ov>.5:conflicts.append({'image_id':r['image_id'],'root_id':r['root_id'],'sample':r['sample'],'camera':r['camera'],'split':r['split'],'IoU':ov,'boxes':boxes.tolist(),'width':r['width'],'height':r['height']})
 save(E/'results/slot_and_GT_overlap.json',{'slots':slot_audit,'second_finger_positive_fraction':slot_audit['2']['positive_centers']/int(y[:,:,1:].sum()),'conflicts':conflicts,'conflicts_per_split':{s:sum(r['split']==s for r in conflicts)for s in ['train','dev']},'GT_pair_test_only':'Perfect overlapping boxes cannot both survive the existing same-class NMS; no NMS/structure/performance changes were made.'})
 summaries=[];points=[];oracle=[];image_stats=[];nms_regressions=0;bins=np.linspace(0,1,21)
 for seed in [0,1]:
  for split in ['train','dev']:
   z=np.load(BASE/f'data/predictions/raw_seed{seed}_{split}.npz',allow_pickle=False);raws=z['raw'];ids=z['image_ids'];assert len(ids)==(256 if split=='train'else 128);records=[];fields={k:[]for k in ['score','tp','image_id','patch','slot','class_id','camera_id','matched_gt','matched_IoU','box']};oc={'seed':seed,'GT_centers':0,'assigned_slot_box_IoU_ge05':0,'assigned_slot_score_lt05':0,'adequate_box_and_score_lt05':0,'meaning':'Train-only oracle at true assigned target centers, not the detector PR/AP population.'}
   for j,i in enumerate(ids):
    r=rows[int(i)];scores,boxes=unpack(raws[j])
    if split=='train':
     targets=toboxes(torch.from_numpy(target[i])).numpy()
     for patch,slot in zip(*np.where(y[i]>0)):
      ov=float(overlaps(boxes[patch,slot],targets[patch,slot][None])[0]);low=bool(scores[patch,slot]<.5);oc['GT_centers']+=1;oc['assigned_slot_box_IoU_ge05']+=int(ov>=.5);oc['assigned_slot_score_lt05']+=int(low);oc['adequate_box_and_score_lt05']+=int(ov>=.5 and low)
    for c,slots in [(0,[0]),(1,[1,2])]:
     if not r['class_valid'][c]:continue
     gt=np.array(r['boxes'][c],np.float64)/[r['width'],r['height'],r['width'],r['height']];b=boxes[:,slots].reshape(-1,4);s=scores[:,slots].reshape(-1);patch=np.repeat(np.arange(1024),len(slots));slot=np.tile(slots,1024);area=(b[:,2]>b[:,0])&(b[:,3]>b[:,1]);order=np.lexsort((slot,patch,-s));order=order[area[order]];keep=nms(b,order);assert np.all(np.diff(s[keep])<=0)
     baseline_keep=reference_nms(b,order[s[order]>=.5]);assert np.array_equal(keep[s[keep]>=.5],baseline_keep);nms_regressions+=1
     tp,matched,miou=match(b[keep],gt);n_gt=len(gt);record={'seed':seed,'split':split,'image_id':int(i),'root_id':r['root_id'],'sample':r['sample'],'camera':r['camera'],'category':CATS[c],'GT':n_gt,'raw_valid_candidates':int(area.sum()),'NMS_kept_candidates':len(keep),'all_score_TP':int(tp.sum()),'never_matched_GT':n_gt-int(tp.sum()),'at05':point(s[keep],tp,n_gt,.5)};image_stats.append(record)
     vals={'score':s[keep],'tp':tp,'image_id':np.full(len(keep),i,np.int32),'patch':patch[keep].astype(np.int16),'slot':slot[keep].astype(np.int8),'class_id':np.full(len(keep),c,np.int8),'camera_id':np.full(len(keep),0 if r['camera']=='static'else 1,np.int8),'matched_gt':matched,'matched_IoU':miou,'box':b[keep]}
     for k,v in vals.items():fields[k].append(v)
   if split=='train':oracle.append(oc)
   arrays={k:np.concatenate(v)for k,v in fields.items()};np.savez_compressed(E/f'data/candidates_seed{seed}_{split}.npz',**arrays)
   for cam in ['all','static','gripper']:
    for cat in ['all']+CATS:
     select=np.ones(len(arrays['score']),bool)
     if cam!='all':select&=arrays['camera_id']==(['static','gripper'].index(cam))
     if cat!='all':select&=arrays['class_id']==CATS.index(cat)
     rr=[r for r in rows if r['split']==split and(cam=='all'or r['camera']==cam)];den=sum(len(r['boxes'][c])for r in rr for c in [0,1]if(cat=='all'or c==CATS.index(cat))and r['class_valid'][c]);a={k:v[select]for k,v in arrays.items()};curve,ap=pr(a['score'],a['tp'],a['image_id'],a['patch'],a['slot'],den);key=f'seed{seed}_{split}_{cam}_{cat}';np.savez_compressed(E/f'data/pr_{key}.npz',**curve);frontier={}
     for precision_floor in [.9,.95]:
      ix=np.where(curve['precision']>=precision_floor)[0]
      if len(ix):
       k=int(ix[np.argmax(curve['recall'][ix])]);frontier[str(precision_floor)]={k2:float(curve[k2][k])for k2 in ['threshold','precision','recall']}
      else:frontier[str(precision_floor)]=None
     n_tp=int(a['tp'].sum());th=np.histogram(a['score'][a['tp']==1],bins)[0];fh=np.histogram(a['score'][a['tp']==0],bins)[0];summary={'seed':seed,'split':split,'camera':cam,'category':cat,'AP50_all_points':ap,'GT':den,'kept_candidates':len(a['score']),'TP_at_all_scores':n_tp,'never_matched_GT':den-n_tp,'recall_ceiling_with_full_normal_decode':n_tp/den,'precision_floor_frontiers':frontier,'hist_bins':bins.tolist(),'TP_hist_counts':th.tolist(),'FP_hist_counts':fh.tolist()};summaries.append(summary)
     assert int(th.sum())+summary['never_matched_GT']==den and int(th.sum()+fh.sum())==len(a['score'])
     for threshold in protocol['descriptive_points']['fixed_score_thresholds']:points.append(dict(seed=seed,split=split,camera=cam,category=cat,GT=den,**point(a['score'],a['tp'],den,threshold)))
   print('PR_COMPLETE',seed,split,'postNMS',len(arrays['score']),flush=True)
 expected={(0,'train'):(220,1,480),(1,'train'):(257,6,443),(0,'dev'):(80,9,274),(1,'dev'):(81,20,273)}
 for q in points:
  if q['camera']=='all'and q['category']=='all'and q['threshold']==.5:assert tuple(q[k]for k in ['TP','FP','FN'])==expected[(q['seed'],q['split'])]
 save(E/'results/summary.json',{'utc':now(),'seconds':time.perf_counter()-start,'AP_definition':protocol['AP'],'rows':summaries});save(E/'results/fixed_threshold_points.json',points);save(E/'results/oracle_GT_center_train_only.json',oracle);save(E/'results/per_image_counts.json',image_stats)
 for name,data in [('threshold_points',points),('ap_summary',[{k:v for k,v in q.items()if k not in ['hist_bins','TP_hist_counts','FP_hist_counts','precision_floor_frontiers']}for q in summaries])]:
  with (R/f'{name}.csv').open('w')as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 assert all(sha(p)==h for p,h in read(E/'configs/input_frozen.json')['sha256'].items());save(E/'results/access_and_checks.json',{'utc':now(),'old_files_opened':sorted(opened),'old_file_reads_allowlisted':True,'D_file_reads':0,'aggregate_all_split_file_reads':0,'old_writes':0,'model_weight_reads':0,'new_inference':0,'training_updates':0,'NMS_prefix_matches_original_05':nms_regressions,'all_histogram_denominators_verified':True,'analytic_AP_missing_GT_and_tie_guards':True,'AB_C_05_counts_match_previous_formal_results':True,'input_hashes_unchanged':True});print('ABC_PR_DIAGNOSTIC_COMPLETE',flush=True)
if __name__=='__main__':main()
