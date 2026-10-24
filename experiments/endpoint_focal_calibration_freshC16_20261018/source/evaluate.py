from common import *
from pr_engine import unpack,nms,reference_nms,match,pr
import csv,time

def metrics(tp,fp,gt):
 return {'TP':int(tp),'FP':int(fp),'FN':int(gt-tp),'precision':float(tp/(tp+fp))if tp+fp else None,'recall':float(tp/gt)if gt else None,'F1':float(2*tp/(gt+tp+fp))if gt+tp+fp else None}
def interval(x):
 x=np.asarray(x);good=x[np.isfinite(x)];return {'lo':float(np.quantile(good,.025))if len(good)else None,'hi':float(np.quantile(good,.975))if len(good)else None,'valid_replicates':len(good),'replicates':len(x)}
def bootstrap_counts(a,sel,gtroot,weights):
 rt=np.bincount(a['root_idx'][sel],weights=a['tp'][sel],minlength=16);rn=np.bincount(a['root_idx'][sel],minlength=16);bt=weights@rt;bn=weights@rn;bg=weights@gtroot
 with np.errstate(invalid='ignore',divide='ignore'):return {'precision':bt/bn,'recall':bt/bg,'F1':2*bt/(bg+bn)}
def boot_ap(a,gtroot,weights):
 # Match and NMS are per image: repeated roots are independent copies, so multiplicities suffice.
 order=np.lexsort((a['slot'],a['patch'],a['image_id'],-a['score']));sc=a['score'][order];tp=a['tp'][order];root=a['root_idx'][order];ends=np.r_[np.flatnonzero(sc[:-1]!=sc[1:]),len(sc)-1];gstart=np.r_[0,ends[:-1]+1];increments=np.add.reduceat(tp.astype(np.int64),gstart);pick=ends[increments>0]
 counts=np.zeros((16,len(pick)),np.int64);totals=counts.copy()
 for r in range(16):counts[r]=np.cumsum(tp*(root==r),dtype=np.int64)[pick];totals[r]=np.cumsum(root==r,dtype=np.int64)[pick]
 ct=weights@counts;nt=weights@totals;gt=weights@gtroot
 with np.errstate(invalid='ignore',divide='ignore'):p=np.divide(ct,nt,out=np.zeros_like(ct,dtype=float),where=nt>0);r=ct/gt[:,None]
 envelope=np.maximum.accumulate(p[:,::-1],axis=1)[:,::-1];ap=np.sum(np.diff(np.c_[np.zeros(len(weights)),r],axis=1)*envelope,axis=1)
 return ap

def main():
 install_guard([]);assert read(E/'results/data_gate.json')['passed'];assert not(E/'results/evaluation_summary.json').exists();assert all(sha(p)==h for p,h in read(E/'results/predictions_frozen.json')['sha256'].items());assert all(sha(p)==h for p,h in read(E/'configs/precollection_frozen.json')['sha256'].items());start=time.perf_counter();rows=read(E/'data/public/images.json');anns=read(E/'data/annotation_private/labels.json');rootnames=[r['root_id']for r in read(E/'configs/protocol.json')['roots']];cal=read(E/'configs/calibration_locked.json')['rows'];dest=E/'data/evaluation';dest.mkdir();rng=np.random.default_rng(261019);draw=rng.integers(0,16,size=(2000,16));weights=np.array([np.bincount(r,minlength=16)for r in draw]);np.save(dest/'bootstrap_root_multiplicities.npy',weights);allrows=[];rootrows=[];paired=[];nchecks=0;reconstructed=0
 for seed in [0,1]:
  z=np.load(E/f'data/predictions/raw_seed{seed}_fresh.npz');fields={k:[]for k in ['score','tp','image_id','root_idx','patch','slot','class_id','camera_id','matched_gt','box']}
  for i,raw in zip(z['image_ids'],z['raw']):
   r,a=rows[int(i)],anns[int(i)];scores,boxes=unpack(raw)
   for cid,slots in [(0,[0]),(1,[1,2])]:
    if not a['class_valid'][cid]:continue
    gt=np.array(a['boxes'][cid],float)/[r['width'],r['height'],r['width'],r['height']];b=boxes[:,slots].reshape(-1,4);s=scores[:,slots].reshape(-1);patch=np.repeat(np.arange(1024),len(slots));slot=np.tile(slots,1024);area=(b[:,2]>b[:,0])&(b[:,3]>b[:,1]);order=np.lexsort((slot,patch,-s));order=order[area[order]];keep=nms(b,order)
    threshold=next(q['threshold']for q in cal if q['seed']==seed and q['category']==CATS[cid])
    for t in [.5]+([]if threshold is None else[threshold]):assert np.array_equal(keep[s[keep]>=t],reference_nms(b,order[s[order]>=t]));nchecks+=1
    tp,matched,miou=match(b[keep],gt);vals={'score':s[keep],'tp':tp,'image_id':np.full(len(keep),i,np.int32),'root_idx':np.full(len(keep),rootnames.index(r['root_id']),np.int8),'patch':patch[keep],'slot':slot[keep],'class_id':np.full(len(keep),cid,np.int8),'camera_id':np.full(len(keep),0 if r['camera']=='static'else 1,np.int8),'matched_gt':matched,'box':b[keep]}
    for k,v in vals.items():fields[k].append(v)
  arr={k:np.concatenate(v)for k,v in fields.items()};np.savez_compressed(dest/f'candidates_seed{seed}.npz',**arr);thresholds=np.array([next(q['threshold']if q['threshold']is not None else np.inf for q in cal if q['seed']==seed and q['category']==c)for c in CATS])
  for cam in ['all','static','gripper']:
   for cat in ['all']+CATS:
    choose=np.ones(len(arr['score']),bool)
    if cam!='all':choose &= arr['camera_id']==['static','gripper'].index(cam)
    if cat!='all':choose &= arr['class_id']==CATS.index(cat)
    a={k:v[choose]for k,v in arr.items()};gtroot=np.zeros(16,np.int64)
    for r,label in zip(rows,anns):
     if cam!='all'and r['camera']!=cam:continue
     gtroot[rootnames.index(r['root_id'])]+=sum(len(label['boxes'][c])for c in [0,1]if label['class_valid'][c]and(cat=='all'or CATS[c]==cat))
    gt=int(gtroot.sum());curve,ap=pr(a['score'],a['tp'],a['image_id'],a['patch'],a['slot'],gt);np.savez_compressed(dest/f'pr_seed{seed}_{cam}_{cat}.npz',**curve);bap=boot_ap(a,gtroot,np.vstack([np.ones((1,16),int),weights]));assert abs(bap[0]-ap)<1e-12;reconstructed+=1;bp={}
    for point in ['fixed0.5','calibrated']:
     sel=a['score']>=(.5 if point=='fixed0.5'else thresholds[a['class_id']]);tp=int(a['tp'][sel].sum());fp=int(sel.sum())-tp;bs=bootstrap_counts(a,sel,gtroot,weights);bp[point]=bs;record=dict(seed=seed,camera=cam,category=cat,workpoint=point,GT=gt,AP50=ap,AP50_CI95=interval(bap[1:]),**metrics(tp,fp,gt),CI95={k:interval(v)for k,v in bs.items()});allrows.append(record)
     for root,rootname in enumerate(rootnames):
      scope=a['root_idx']==root;rp=sel&scope;rcurve,rap=pr(a['score'][scope],a['tp'][scope],a['image_id'][scope],a['patch'][scope],a['slot'][scope],int(gtroot[root]))if gtroot[root]>0 and scope.any()else({},None);rootrows.append(dict(seed=seed,camera=cam,category=cat,workpoint=point,root_id=rootname,GT=int(gtroot[root]),AP50=rap,**metrics(int(a['tp'][rp].sum()),int(rp.sum()-a['tp'][rp].sum()),int(gtroot[root]))))
    paired.append({'seed':seed,'camera':cam,'category':cat,'difference':'calibrated minus fixed0.5','CI95':{k:interval(bp['calibrated'][k]-bp['fixed0.5'][k])for k in ['precision','recall','F1']},'AP_difference':0.})
  print('NEW_C_SCORED_ONCE',seed,flush=True)
 save(E/'results/evaluation_summary.json',{'utc':now(),'rows':allrows,'new_roots':16,'GT_total':allrows[0]['GT'],'AP_independent_of_operating_threshold':True,'scoring_invocations':1,'model_inference_passes_per_seed':1,'elapsed_seconds':time.perf_counter()-start,'D_reads':0});save(E/'results/root_metrics.json',rootrows);save(E/'results/paired_bootstrap.json',paired);save(E/'results/evaluation_checks.json',{'utc':now(),'normal_NMS_threshold_prefix_checks':nchecks,'compressed_AP_matches_full_PR':reconstructed,'bootstrap_replicates':2000,'root_pairing_shared_across_all_conditions':True,'new_threshold_or_model_selection':False})
 for filename,rr in [('metrics.csv',allrows),('root_metrics.csv',rootrows)]:
  with(R/filename).open('w')as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
 for q in allrows:
  if q['camera']=='all':print(q,flush=True)
if __name__=='__main__':main()
