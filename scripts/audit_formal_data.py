"""Validate frozen formal quotas and channels before any formal training."""
from pathlib import Path
import json,hashlib,collections,datetime,os
import numpy as np
from wm_symbolic.root_registry import active_root_directory
P=Path('/home/robot/research/projects/wm_symbolic');manifest_path=P/'data/data_manifest.json';raw=manifest_path.read_bytes();m=json.loads(raw);rows=m['records'];expected={'train':(240,4320,'AB'),'dev':(30,360,'C'),'calibration':(30,360,'C'),'test':(80,960,'D')}
errors=[];by_episode=collections.defaultdict(list);by_root=collections.defaultdict(list);report={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'manifest_sha256':hashlib.sha256(raw).hexdigest(),'split_counts':{},'split_roots':{},'errors':errors}
events=['red_moved','blue_moved','pink_moved','led_was_off','led_was_on','light_was_off','light_was_on']
if m.get('event_names')!=events:errors.append('v2 process event interface missing')
skills={s['skill']:s['family'] for s in json.loads((P/'configs/skills.json').read_text())};budgets=json.loads((P/'configs/executor_budgets_pilot.json').read_text())['steps'];styles=collections.Counter();success=collections.defaultdict(lambda:[0,0]);unknown=collections.Counter();keys=set()
for r in rows:
 key=(r['episode_id'],r['step'])
 if key in keys:errors.append('Duplicate transition '+str(key))
 keys.add(key);by_episode[r['episode_id']].append(r);by_root[r['root_id']].append(r);styles[r['collection_style']]+=1
 if r.get('label_version')!='calibrated_geometry_and_process_v2':errors.append('Unfrozen labels '+str(key))
 if r['executor']['budget_steps']!=budgets[skills[r['skill']]]:errors.append('Skill budget differs '+str(key))
 for name in ['obs_before','obs_after','low_level_log','labels_path','raw_evaluator_trace']:
  if not Path(r.get(name,'/missing')).is_file():errors.append('Missing '+name+' '+str(key))
 label=json.loads(Path(r['labels_path']).read_text())
 if len(label['events'])!=7 or len(label['events_valid'])!=7:errors.append('Wrong event shape '+str(key))
 if bool(label['success'])!=r['success']:errors.append('Success label changed '+str(key))
 for side in ['before','after']:
  for i,valid in enumerate(label[side]['categorical_valid']):unknown[f'categorical_{i}']+=not valid
  for i,valid in enumerate(label[side]['binary_valid']):unknown[f'binary_{i}']+=not valid
 for i,valid in enumerate(label['events_valid']):unknown[f'event_{i}']+=not valid
 success[r['split']+':'+r['skill']][0]+=int(r['success']);success[r['split']+':'+r['skill']][1]+=1
for split,(roots,n,envs) in expected.items():
 selected=[r for r in rows if r['split']==split];ids=sorted({r['root_id'] for r in selected});report['split_counts'][split]=len(selected);report['split_roots'][split]=ids
 if len(selected)!=n or len(ids)!=roots:errors.append('Root/transition quota mismatch '+split)
 if any(r['environment'] not in envs for r in selected):errors.append('Environment split mismatch '+split)
for eid,records in by_episode.items():
 if sorted(r['step'] for r in records)!=list(range(6)):errors.append('Incomplete six-skill episode '+eid)
 if len({r['root_id'] for r in records})!=1:errors.append('Episode root changed '+eid)
hash_roots=collections.defaultdict(list)
for root,records in by_root.items():
 splits={r['split'] for r in records};environments={r['environment'] for r in records}
 if len(splits)!=1 or len(environments)!=1:errors.append('Root leakage '+root)
 split=records[0]['split'];branches={r['episode_id'] for r in records}
 if len(branches)!=(3 if split=='train' else 2):errors.append('Root branch quota mismatch '+root)
 directory=active_root_directory(P,root,rows);identity=json.loads((directory/'identity.json').read_text())
 if identity['split']!=split:errors.append('Root identity mismatch '+root)
 root_hash=json.loads((directory/'public_root_hash.json').read_text())['sha256'];hash_roots[root_hash].append(root)
 if len(root_hash)!=64:errors.append('Invalid public root hash '+root)
 if not (directory/'full_state_evaluator_only.json').is_file():errors.append('Missing evaluator root '+root)
 if not (directory/'native_config.yaml').is_file():errors.append('Missing reset configuration '+root)
duplicates={k:v for k,v in hash_roots.items() if len(v)>1}
if duplicates:errors.append('Duplicate physical public root hashes')
index=json.loads((P/'data/features/dinov2_v1/index.json').read_text());wanted={(eid,t) for eid in by_episode for t in range(7)};actual=[(x['episode_id'],x['step']) for x in index['items']]
if len(actual)!=len(set(actual)) or set(actual)!=wanted:errors.append('Feature cache coverage mismatch')
if not index.get('frozen') or not index.get('only_public_inputs'):errors.append('Feature cache metadata not frozen/public')
valid_frame_keys={'rgb_static','rgb_gripper','depth_static','depth_gripper','seg_static','seg_gripper','robot_obs'}|{f'calib_{camera}_{name}' for camera in ['static','gripper'] for name in ['width','height','fov','nearval','farval','viewMatrix','projectionMatrix']}
frame_keysets=collections.Counter();feature_bytes=0
for item in index['items']:
 path=Path(item['path']);feature_bytes+=path.stat().st_size
 with np.load(path) as z:
  if z['tokens'].shape!=(512,384) or z['geometry'].shape!=(512,8) or z['valid'].shape!=(512,) or z['robot'].shape!=(15,):errors.append('Feature shape mismatch '+str(path))
  if not all(np.isfinite(z[k]).all() for k in ['tokens','geometry','robot','reconstruction']):errors.append('Nonfinite feature '+str(path))
 if len(item['public_inputs'])!=4:errors.append('Frame history length differs '+str(path))
 # Inspect every actual public boundary frame keyset; hidden arrays must not enter model cache.
 for frame in item['public_inputs']:
  with np.load(frame) as z:frame_keysets[','.join(sorted(z.files))]+=1
for names in frame_keysets:
 if not set(names.split(','))<=valid_frame_keys:errors.append('Non-public field in model frame '+names)
report.update(total_transitions=len(rows),total_roots=len(by_root),total_episodes=len(by_episode),feature_boundaries=len(actual),feature_bytes=feature_bytes,
 collection_styles=dict(styles),actual_success_by_split_skill=dict(success),label_unknown_counts=dict(unknown),duplicate_root_hashes=duplicates,public_frame_keysets=dict(frame_keysets),
 target_role='Evaluator labels are supervision/scoring only; no per-task goals or actual success are model inputs',test_policy='D records frozen; this audit checks integrity, quotas and channels only, without model selection',passed=not errors)
out=P/'runs/formal_data_audit.json';temp=out.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2));temp.replace(out)
print('FORMAL_DATA_AUDIT',json.dumps({'passed':report['passed'],'split_counts':report['split_counts'],'roots':len(by_root),'features':len(actual),'errors':errors}),flush=True)
if errors:raise SystemExit(1)
