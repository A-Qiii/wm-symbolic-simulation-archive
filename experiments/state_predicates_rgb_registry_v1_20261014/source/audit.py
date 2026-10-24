from pathlib import Path
import json,hashlib,argparse,datetime
import numpy as np
P=Path('/home/robot/research/projects/wm_symbolic');E=P/'experiments/state_predicates_rgb_registry_v1_20261014';OLD=P/'experiments/state_predicates_long_pairs_20261014';B=P/'data/predicate_long_pairs_20261014';C=['red','blue','pink']
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def score(pred,gt,blocks):
 area=int(pred.sum());n=int(gt.sum());inter=int((pred&gt).sum());correct=bool(area>=2 and inter/area>.5);iou=inter/max(1,int((pred|gt).sum()));bg=bool(area>=2 and int((pred&blocks).sum())/area<=.5)
 return {'detected':area>=2,'pred_pixels':area,'GT_pixels':n,'intersection':inter,'precision':inter/area if area else None,'IoU':iou,'correct_identity':correct,'background_false_positive':bg,'wrong_identity':area>=2 and not correct,'visible_miss':n>=4 and not correct,'occluded_hallucination':n==0 and area>=2}
def summarize(rows,version):
 out={}
 for c in C:
  rr=[r for r in rows if r['color']==c];clear=[r for r in rr if r['GT_pixels']>=4];occ=[r for r in rr if r['GT_pixels']==0];good=[r[version]['IoU']for r in clear if r[version]['correct_identity']];out[c]={'opportunities':len(rr),'clearly_visible':len(clear),'correct_visible_detections':sum(r[version]['correct_identity']for r in clear),'visible_recall':sum(r[version]['correct_identity']for r in clear)/len(clear)if clear else None,'tiny_1to3px':sum(0<r['GT_pixels']<4 for r in rr),'fully_occluded':len(occ),'occluded_hallucinations':sum(r[version]['detected']for r in occ),'background_false_positives':sum(r[version]['background_false_positive']for r in rr),'wrong_identity_detections':sum(r[version]['wrong_identity']for r in rr),'all_detections':sum(r[version]['detected']for r in rr),'visible_IoU_quantiles':[float(x)for x in np.quantile(good,[0,.1,.5,.9,1])]if good else [],'visible_IoU_at_least_half':sum(r[version]['correct_identity']and r[version]['IoU']>=.5 for r in clear)}
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['train','dev'],required=True);args=ap.parse_args();meta=read(E/'data'/f'windows_{args.split}.json');old={r['window_id']:r for r in read(OLD/'results/cache_full.json')['audits']};legacy_missing=read(OLD/'results/registry_diagnostic_full.json')['details'];rows=[];window_audit=[];missing=[]
 for r in meta:
  wid=r['window_id'];aud=old[wid];uids={c:aud['registry_audit'][c]['private_uid']for c in C};win=[]
  with np.load(r['rgb_regions_path'])as masks:
   assert sha(r['rgb_regions_path'])==r['rgb_regions_sha256']
   for ti,(fi,fp)in enumerate(zip(r['history_frame_indices'],r['source_public_frames'])):
    # Evaluator only: native pixel masks are opened AFTER RGB-only regions exist.
    with np.load(fp)as z:
     for cam in ['static','gripper']:
      seg=z['seg_'+cam];base=seg&((1<<24)-1);blocks=(seg>=0)&np.isin(base,list(uids.values()))
      for ci,c in enumerate(C):
       gt=(seg>=0)&(base==uids[c]);oid=aud['registry_audit'][c]['registered_public_uid'];opred=seg==oid if oid is not None else np.zeros_like(gt);opred=opred if opred.sum()>=2 else np.zeros_like(gt);nscore=score(masks[cam][ti,ci],gt,blocks);oscore=score(opred,gt,blocks);row={'window_id':wid,'root_id':r['root_id'],'split':r['split'],'frame':fi,'camera':cam,'color':c,'GT_pixels':int(gt.sum()),'original':oscore,'rgb_v1':nscore};rows.append(row);win.append(row)
  for c in C:
   rr=[q for q in win if q['color']==c];maximum=max(q['GT_pixels']for q in rr);anchor=[q for q in rr if q['frame']==r['anchor_frame']];current=max(q['GT_pixels']for q in anchor);kind='fully_occluded_all8_images'if maximum==0 else'tiny_or_marginal_only'if maximum<4 else'visible_in_past_currently_occluded'if current==0 else'visible_in_window';entry={'window_id':wid,'root_id':r['root_id'],'color':c,'visibility_class':kind,'GT_max_pixels_any_image':maximum,'GT_max_pixels_anchor':current,'truth_at_anchor':{'held':aud['all7_targets']['y'][C.index(c)],'in':aud['all7_targets']['y'][3+C.index(c)]},'new_any_correct_detection':any(q['rgb_v1']['correct_identity']for q in rr),'new_wrong_identity_views':sum(q['rgb_v1']['wrong_identity']for q in rr),'new_missing_all8_images':not any(q['rgb_v1']['detected']for q in rr)};window_audit.append(entry)
   if any(q['window_id']==wid and q['color']==c and q['missing']for q in legacy_missing):missing.append(entry)
 new=summarize(rows,'rgb_v1');original=summarize(rows,'original');gate=all(q['visible_recall']is not None and q['visible_recall']>=.95 and q['background_false_positives']==0 and q['occluded_hallucinations']==0 and q['wrong_identity_detections']==0 for q in new.values());result={'split':args.split,'window_count':len(meta),'view_colour_opportunities':len(rows),'legacy_missing_windows':missing,'original':original,'rgb_v1':new,'registration_gate_passed':gate,'all_29_original_missing_are_fully_occluded_in_this_subset':all(q['visibility_class']=='fully_occluded_all8_images'for q in missing),'new_wrong_identity_windows':sorted({q['window_id']for q in rows if q['rgb_v1']['wrong_identity']}),'new_visible_miss_windows':sorted({q['window_id']for q in rows if q['rgb_v1']['visible_miss']}),'window_audit':window_audit,'details':rows,'evaluator_native_masks_never_used_to_construct_features':True};save(E/'results'/f'registration_{args.split}.json',result)
 if args.split=='train'and gate:
  freeze={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'AB_registration_gate_passed':True,'rule_sha256':sha(E/'source/registry.py'),'AB_results_sha256':sha(E/'results/registration_train.json'),'AB_public_windows_sha256':sha(E/'data/windows_train.json'),'no_C_pixels_used_for_rule_development':True,'no_more_rule_edits':True};save(E/'configs/frozen_before_C.json',freeze);(E/'configs/frozen_before_C.json').chmod(0o444)
 print(json.dumps({k:v for k,v in result.items()if k not in ['details','window_audit','legacy_missing_windows']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
