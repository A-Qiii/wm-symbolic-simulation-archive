from common import *
from pr_engine import unpack,nms,reference_nms,match,point,pr
import csv
def main():
 assert not(E/'results/evaluation_summary.json').exists();install_guard([]);assert all(sha(p)==h for p,h in read(E/'results/predictions_frozen.json')['sha256'].items());rows=read(E/'data/ABC_labels_metadata.json');baseline=read(E/'data/baseline/summary.json')['rows'];basepoints=read(E/'data/baseline/fixed_threshold_points.json');summaries=[];points=[];image_stats=[];nms_checks=0;bins=np.linspace(0,1,21)
 for seed in [0,1]:
  for split in ['train','dev']:
   z=np.load(E/f'data/predictions/raw_seed{seed}_{split}.npz');fields={k:[]for k in ['score','tp','image_id','patch','slot','class_id','camera_id','matched_gt','matched_IoU','box']}
   for i,raw in zip(z['image_ids'],z['raw']):
    r=rows[int(i)];assert r['split']==split;scores,boxes=unpack(raw)
    for c,slots in [(0,[0]),(1,[1,2])]:
     if not r['class_valid'][c]:continue
     gt=np.array(r['boxes'][c],float)/[r['width'],r['height'],r['width'],r['height']];b=boxes[:,slots].reshape(-1,4);s=scores[:,slots].reshape(-1);patch=np.repeat(np.arange(1024),len(slots));slot=np.tile(slots,1024);area=(b[:,2]>b[:,0])&(b[:,3]>b[:,1]);order=np.lexsort((slot,patch,-s));order=order[area[order]];keep=nms(b,order);assert np.array_equal(keep[s[keep]>=.5],reference_nms(b,order[s[order]>=.5]));nms_checks+=1;tp,matched,miou=match(b[keep],gt);image_stats.append({'seed':seed,'split':split,'image_id':int(i),'root_id':r['root_id'],'sample':r['sample'],'camera':r['camera'],'category':CATS[c],'GT':len(gt),'kept_candidates':len(keep),'never_matched_GT':len(gt)-int(tp.sum()),'at05':point(s[keep],tp,len(gt),.5)})
     values={'score':s[keep],'tp':tp,'image_id':np.full(len(keep),i,np.int32),'patch':patch[keep].astype(np.int16),'slot':slot[keep].astype(np.int8),'class_id':np.full(len(keep),c,np.int8),'camera_id':np.full(len(keep),0 if r['camera']=='static'else 1,np.int8),'matched_gt':matched,'matched_IoU':miou,'box':b[keep]}
     for k,v in values.items():fields[k].append(v)
   arrays={k:np.concatenate(v)for k,v in fields.items()};np.savez_compressed(E/f'data/evaluation/candidates_seed{seed}_{split}.npz',**arrays)
   for cam in ['all','static','gripper']:
    for cat in ['all']+CATS:
     sel=np.ones(len(arrays['score']),bool)
     if cam!='all':sel&=arrays['camera_id']==['static','gripper'].index(cam)
     if cat!='all':sel&=arrays['class_id']==CATS.index(cat)
     a={k:v[sel]for k,v in arrays.items()};rr=[r for r in rows if r['split']==split and(cam=='all'or r['camera']==cam)];den=sum(len(r['boxes'][c])for r in rr for c in [0,1]if r['class_valid'][c]and(cat=='all'or c==CATS.index(cat)))
     if len(a['score']):curve,ap=pr(a['score'],a['tp'],a['image_id'],a['patch'],a['slot'],den)
     else:curve={k:np.array([],np.float64)for k in ['threshold','precision','recall','TP','FP','precision_envelope']};ap=0.
     np.savez_compressed(E/f'data/evaluation/pr_seed{seed}_{split}_{cam}_{cat}.npz',**curve);nt=int(a['tp'].sum());summary={'seed':seed,'split':split,'camera':cam,'category':cat,'AP50_all_points':ap,'GT':den,'kept_candidates':len(a['score']),'TP_at_all_scores':nt,'never_matched_GT':den-nt,'recall_ceiling_with_full_normal_decode':nt/den,'hist_bins':bins.tolist(),'TP_hist_counts':np.histogram(a['score'][a['tp']==1],bins)[0].tolist(),'FP_hist_counts':np.histogram(a['score'][a['tp']==0],bins)[0].tolist()};summaries.append(summary)
     for threshold in [.5,.3,.1]:points.append(dict(seed=seed,split=split,camera=cam,category=cat,GT=den,**point(a['score'],a['tp'],den,threshold)))
   print('MIXED_AP_DONE',seed,split,flush=True)
 comparisons=[]
 for q in summaries:
  key={k:q[k]for k in ['seed','split','camera','category']};b=next(x for x in baseline if all(x[k]==v for k,v in key.items()));p=next(x for x in points if x['threshold']==.5 and all(x[k]==v for k,v in key.items()));bp=next(x for x in basepoints if x['threshold']==.5 and all(x[k]==v for k,v in key.items()));assert b['GT']==q['GT'];comparisons.append(dict(**key,GT=q['GT'],focal_AP50=b['AP50_all_points'],mixed_AP50=q['AP50_all_points'],delta_AP50=q['AP50_all_points']-b['AP50_all_points'],focal_precision05=bp['precision'],mixed_precision05=p['precision'],focal_recall05=bp['recall'],mixed_recall05=p['recall'],focal_F1_05=bp['F1'],mixed_F1_05=p['F1'],focal_TP05=bp['TP'],mixed_TP05=p['TP'],focal_FP05=bp['FP'],mixed_FP05=p['FP'],focal_FN05=bp['FN'],mixed_FN05=p['FN']))
 directions=[]
 for split in ['train','dev']:
  for cam in ['all','static','gripper']:
   for cat in ['all']+CATS:
    q=sorted([x for x in comparisons if x['split']==split and x['camera']==cam and x['category']==cat],key=lambda x:x['seed']);d=[x['delta_AP50']for x in q];direction='both_improve'if min(d)>1e-12 else('both_decline'if max(d)<-1e-12 else'mixed_or_tied');directions.append({'split':split,'camera':cam,'category':cat,'delta_AP50_by_seed':d,'direction':direction})
 save(E/'results/evaluation_summary.json',{'utc':now(),'rows':summaries,'AP_definition':'same all-points AP50, no score floor, normal NMS, all valid GT','D_reads':0,'scoring_passes':1});save(E/'results/fixed_threshold_points.json',points);save(E/'results/per_image_counts.json',image_stats);save(E/'results/comparison.json',comparisons);save(E/'results/seed_consistency.json',directions);save(E/'results/evaluation_checks.json',{'utc':now(),'NMS_05_prefix_reference_checks':nms_checks,'baseline_reused_without_new_inference':True,'D_data_or_metrics_reads':0,'threshold_search_or_selection':False,'raw_predictions_hashes_verified':True})
 with(R/'comparison.csv').open('w')as f:w=csv.DictWriter(f,fieldnames=list(comparisons[0]));w.writeheader();w.writerows(comparisons)
 for q in comparisons:
  if q['camera']=='all':print(q,flush=True)
if __name__=='__main__':main()
