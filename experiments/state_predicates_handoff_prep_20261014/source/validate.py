from pathlib import Path
import json,hashlib,datetime
P=Path('/home/robot/research/projects/wm_symbolic');E=P/'experiments/state_predicates_handoff_prep_20261014'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def main():
 assert not(E/'results/preparation_validation.json').exists();meta=read(E/'data/metadata_long_history_v1.json');maps=read(E/'results/metadata_repair_mapping.json');v=read(E/'results/metadata_repair_validation.json');assert len(meta)==len(maps)==54 and sum(x['split']=='train'for x in meta)==36
 assert v['corrected_metadata_sha256']==sha(E/'data/metadata_long_history_v1.json')and v['maximum_tensor_error']==0
 for q,m in zip(meta,maps):
  assert q['sample_id']==m['sample_id'];assert q['history_frame_indices']==[q['anchor_frame']-x for x in [63,42,21,0]];assert sha(q['feature_path'])==q['feature_sha256'];assert m['mask_unchanged']and m['anchor_label_matches_saved_y']
  for k,change in m['changes'].items():assert q[k]==change['after']
 pairs=[json.loads(x)for x in(E/'data/one_step_short_train_dev_v1.jsonl').read_text().splitlines()];assert len(pairs)==960 and sum(x['partition']=='train'for x in pairs)==720 and sum(x['partition']=='dev'for x in pairs)==240
 for x in pairs:
  assert x['source']=='core1200'and x['partition']!='prior_D'and x['short_history_both_cached'];assert all(len(x[s]['recorded_short_public_paths'])==4 for s in ['before','after'])
  for k in ['before_feature_path','after_feature_path']:assert sha(x[k])==x[k+'_sha256']
 old=read(E/'results/source_hashes.json');count=0
 for p,h in old.items():
  if not Path(p).is_relative_to(E):assert sha(p)==h,p;count+=1
 for p,h in read(E/'configs/preparation_scope.json')['before_sha256'].items():assert sha(p)==h,p
 preserved=[]
 for mp in [P/'experiments/state_predicates_interface_acceptance_20261014/delivery_manifest.json',P/'data/predicate_interface_acceptance_20261014/delivery_manifest.json',P/'experiments/state_predicates_temporal_20261014/delivery_manifest.json',P/'experiments/state_predicates_semantics_audit_20261014/delivery_manifest.json']:
  if not mp.exists():continue
  m=read(mp);root=Path(m.get('root',mp.parent));entries=m.get('entries',[])
  for q in entries:
   p=root/q['path'];assert p.stat().st_size==q['bytes']and sha(p)==q['sha256'],str(p)
  preserved.append({'manifest':str(mp),'sha256':sha(mp),'files_verified':len(entries)})
 inv=read(E/'data/macro_transition_history_inventory_v2.json');assert len(inv)==1278
 assert not any(x['before']['complete_2p1s']and x['after']['complete_2p1s']for x in inv if x['partition']in ['train','dev'])
 both=[x for x in inv if x['before']['complete_2p1s']and x['after']['complete_2p1s']];assert len(both)==24 and all(x['partition']=='prior_D'and not x['model_selection_allowed']for x in both)
 pub=read(E/'results/public_file_checks.json')
 for p,r in pub.items():assert sha(p)==r['sha256']and r['public_keys_only']
 result={'passed':True,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'metadata_repaired_and_exactly_checked':54,'train_dev':{'train':36,'dev':18},'input_scalar_values_compared':v['matched_objects_context_robot_scalar_values'],'maximum_input_error':0,'prepared_short_one_step_pairs':960,'macro_calls_inventoried':1278,'long_pairs_train':0,'long_pairs_C':0,'long_pairs_previous_D':24,'public_files_read_and_checked':len(pub),'old_source_files_rehashed':count,'old_manifests_preserved':preserved,'training_updates':0,'model_forward_calls':0,'collection_or_replay_calls':0,'candidate_or_threshold_changes':0,'prior_D_used_to_select_model':False,'new_test_and_training_are_proposals_only':True}
 save(E/'results/preparation_validation.json',result);files=[]
 for p in sorted(E.rglob('*')):
  if p.is_file()and p.name!='delivery_manifest.json':files.append({'path':str(p.relative_to(E)),'bytes':p.stat().st_size,'sha256':sha(p)})
 save(E/'delivery_manifest.json',{'root':str(E),'algorithm':'SHA256','files':len(files),'bytes':sum(x['bytes']for x in files),'entries':files});print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
