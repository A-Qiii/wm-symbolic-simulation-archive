from common import *
def main():
 assert not(E/'results/final_verification.json').exists();rows=read(E/'data/public/images.json');anns=read(E/'data/annotation_private/labels.json');matches=read(E/'results/matches.json');frozen_checked={}
 for name in ['configs/precollection_frozen.json','configs/feature_code_frozen.json','configs/data_frozen.json','configs/training_frozen.json','results/predictions_frozen.json']:
  manifest=read(E/name)['sha256'];bad=[p for p,h in manifest.items()if not Path(p).exists()or sha(p)!=h];assert not bad,(name,bad);frozen_checked[name]=len(manifest)
 snapshots=0;captures=0
 for r in rows[::2]:
  d=E/'data/annotation_private'/r['root_id']/f's{r["sample"]:02d}';assert read(d/'snapshot_before.json')==read(d/'snapshot_after.json');snapshots+=1
  for k in [0,1]:
   a=np.load(d/f'call{k:02d}.npz',allow_pickle=False);b=np.load(d/f'call{k+2:02d}.npz',allow_pickle=False);assert np.array_equal(a['rgb'],b['rgb'])and np.array_equal(a['depth_buffer'],b['depth_buffer']);pub=np.load(r['path'],allow_pickle=False);assert np.array_equal(pub[['rgb_static','rgb_gripper'][k]],a['rgb']);assert read(d/f'call{k:02d}.json')['tick']==read(d/f'call{k+2:02d}.json')['tick'];captures+=1
 for rec in matches:
  a=anns[rec['image_id']];c=CATS.index(rec['category']);assert a['class_valid'][c];assert rec['TP']+rec['FN']==len(a['boxes'][c]);assert rec['TP']==len(rec['matches']);assert len({m['gt']for m in rec['matches']})==rec['TP'];assert all(m['IoU']>=.5 for m in rec['matches'])
 metric=read(E/'results/metrics.json');assert metric['evaluation_passes']==1
 for m in metric['metrics']:
  rr=[r for r in matches if r['seed']==m['seed']and r['split']==m['split']and(m['camera']=='all'or r['camera']==m['camera'])and(m['category']=='all'or r['category']==m['category'])]
  for k in ['TP','FP','FN']:assert m[k]==sum(r[k]for r in rr)
  assert abs(m['F1']-2*m['TP']/(2*m['TP']+m['FP']+m['FN']))<1e-12
 oldnames=['endpoint_box_localizer_20261018','endpoint_box_resolution_224_448_20261018','endpoint_box_decoder_all_candidates_20261018','endpoint_box_label_adjudication8_20261018','endpoint_box_user_arbitration_20261018','visual_node_edge_readout_20261018','visual_edge_zero_context_control_20261018','endpoint_rgb_feasibility_20261018','endpoint_simulator_truth_audit_20261018','endpoint_sync_truth_4state_20261018','endpoint_sync_truth_4state_retry_20261018'];old=[]
 for name in oldnames:
  p=P/'experiments'/name/'configs/delivery_manifest.json';d=read(p);bad=[f for f,h in d['sha256'].items()if not Path(f).exists()or sha(f)!=h];assert not bad,(name,bad);old.append({'name':name,'entries_verified':len(d['sha256']),'manifest_sha256':sha(p)})
 save(E/'results/final_verification.json',{'utc':now(),'passed':True,'frozen_boundaries_verified':frozen_checked,'full_snapshot_pairs':snapshots,'raw_flag_on_off_and_public_RGB_pairs':captures,'metric_record_arithmetic_checked':len(matches),'metric_rescoring_or_new_predictions':False,'old_experiments':old,'old_file_entries_unchanged':sum(o['entries_verified']for o in old),'remaining_experimental_failure':'fixed baseline has low train and holdout recall; no additional fits or deployment','technical_failures':0,'collection_retries':0});print('VERIFIED_OLD_ENTRIES',sum(o['entries_verified']for o in old),flush=True)
if __name__=='__main__':main()
