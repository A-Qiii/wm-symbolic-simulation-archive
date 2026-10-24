from pathlib import Path
from collections import Counter
import json,hashlib
P=Path('/home/robot/research/projects/wm_symbolic');E=P/'experiments/state_predicates_handoff_prep_20261014'
def read(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def main():
 assert not(E/'results/history_recovery_counts.json').exists();meta=read(E/'data/macro_transition_history_inventory.json');rows=[]
 for m in meta:
  pre=0;setup=None
  if m['source']=='core1200':
   setup=Path(m['before']['recorded_short_public_paths'][0]).parent
   if setup.name.startswith(('00_','01_','02_','03_')):setup=setup.parent
   setup=setup/'setup_private.json';pre=24+len(read(setup)['controls'])
  r={'id':m['id'],'partition':m['partition'],'source':m['source'],'root_id':m['root_id'],'known_control_steps_before_timeline_zero':pre,'setup_path':str(setup)if setup else None}
  for side in ['before','after']:
   end=m[side];elapsed=end['anchor_control_index']+pre
   if end['complete_2p1s']:kind='already_complete_public'
   elif elapsed<63:kind='insufficient_executed_prehistory_requires_new_trajectory'
   else:kind='missing_public_files_replay_candidate_not_guaranteed'
   r[side]={'kind':kind,'known_elapsed_control_steps':elapsed,'needed_earlier_physical_controls':max(0,63-elapsed),'source_paths':end['long_public_paths']}
  rows.append(r)
 stats=[]
 for part in ['train','dev','prior_D']:
  for source in sorted({x['source']for x in rows if x['partition']==part}):
   rr=[x for x in rows if x['partition']==part and x['source']==source];stats.append({'partition':part,'source':source,'macro_transitions':len(rr),'before':dict(Counter(x['before']['kind']for x in rr)),'after':dict(Counter(x['after']['kind']for x in rr)),'pairs_require_new_prehistory':sum(any(x[s]['kind']=='insufficient_executed_prehistory_requires_new_trajectory'for s in ['before','after'])for x in rr)})
 save(E/'data/history_recovery_map.json',rows);save(E/'results/history_recovery_counts.json',{'counts':stats,'interpretation':'Replay candidates mean enough historical control steps were executed but original public files are missing; future deterministic replay would need fresh equivalence validation and is not authorized/performed now. Insufficient prehistory cannot be repaired by duplicate frames or GT synthesis; adding initial holds changes the trajectory and must be recorded as new collection on an existing root. Counts are transition endpoint occurrences, not independent roots or necessarily unique windows.'})
 pairs=[json.loads(x)for x in(E/'data/one_step_short_train_dev_v1.jsonl').read_text().splitlines()];fs={r[k]for r in pairs for k in ['before_feature_path','after_feature_path']};shortbytes=sum(Path(f).stat().st_size for f in fs);old=read(P/'runs/core1200_v1/training_main/result.json');ident=read(P/'runs/core1200_v1/training_main/identity.json');logs=[json.loads(x)for x in(P/'runs/core1200_v1/training_main/training.jsonl').read_text().splitlines()];latest=logs[-1];accept=read(P/'data/predicate_interface_acceptance_20261014/data_manifest.json');seconds=sum(x['seconds']for x in accept['roots']);frames=sum(x['frames']for x in accept['roots']);manifest=read(P/'data/predicate_interface_acceptance_20261014/delivery_manifest.json');fc=read(P/'experiments/state_predicates_augmented_20261014/results/cache_train_dev.json')
 save(E/'results/compute_evidence.json',{'actual_historical_measurements':{'world_model_parameters':ident['parameters'],'WM_updates':old['updates'],'WM_wall_seconds':old['elapsed_seconds_this_segment'],'WM_training_protocol':read(P/'runs/core1200_v1/protocol.json')['training'],'WM_torch_peak_allocated_bytes':latest['peak_gpu_bytes'],'frozen_feature_windows':fc['windows'],'frozen_feature_cache_seconds':fc['seconds'],'existing_acceptance_frames':frames,'existing_acceptance_collect_seconds':seconds,'existing_acceptance_dataset_bytes':manifest['bytes']},'reusable_short_cache':{'unique_boundary_cache_files':len(fs),'bytes':shortbytes,'transitions_train':720,'transitions_C':240,'no_new_DINO_required':True},'proposal_estimate_not_measurement':{'one_seed_one_step_pilot_updates':6000,'batch_transitions':8,'gradient_accumulation':2,'accumulation':2,'wall_time_minutes_including_evaluation':[20,40],'GPU_budget_GiB':[2,4],'minimum_new_test_roots':12,'minimum_new_test_macro_transitions':48,'new_test_estimated_total_frames':[16000,22000],'new_test_estimated_bytes':[2800000000,3800000000],'new_test_collect_minutes':[5,10],'new_test_feature_and_eval_minutes':[1,5],'three_seed_extension_minutes':[60,120],'strict_long_history_720_train_240_C_extra_boundary_caches':1200,'strict_long_history_cache_estimated_bytes':[430000000,530000000],'new_collection_new_roots_for_full_guide_not_included':True},'caveat':'Estimates extrapolate existing local runs; no benchmark, model forward, collection or training was performed in this preparation. Peakallocated memory is PyTorch allocation, not total VRAM.6000updates are optimization updates, not6000new transitions.'})
 print(json.dumps({'recovery':stats,'existing_cache_bytes':shortbytes,'historical_WM_seconds':old['elapsed_seconds_this_segment']},ensure_ascii=False))
if __name__=='__main__':main()
