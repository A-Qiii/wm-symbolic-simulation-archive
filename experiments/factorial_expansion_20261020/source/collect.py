from common import *
import argparse,fcntl,hydra,traceback,gzip
from omegaconf import OmegaConf
sys.path.insert(0,str(P/'scripts'))
import episode
def intended(root,lab):
 c=lab['categorical'];a,b,z=[COLORS.index(x)for x in root['role_colors']];k=root['state_kind']
 ins=set(np.where(np.array(c[:3])==3)[0]);held=set(np.where(np.array(c[:3])==6)[0])
 pairs={0:(set(),set()),1:(set(),{a}),2:({a},set()),3:({a},{b}),4:({a,b},set()),5:({a,b},{z})}
 return bool((ins,held)==pairs[k]and all(lab['categorical_valid'])and c[3]==(2 if root['intended_drawer_open']else 0))
def audit_episode(root,branch,rows):
 d=S/'data/episodes'/rows[0]['episode_id'];setup=read(d/'setup_private.json');results=[read(Path(r['labels_path']).parent/'result_private.json')for r in rows]
 combo=bool(setup['heldout_combo_controls'] or episode.heldout(setup['actual_initial_labels'])or any(r['heldout_combo_controls']or r['heldout_combo_boundary']for r in results))
 public_equal=True
 if branch==2:
  d1=S/'data/episodes'/f"{root['root_id']}_b1"
  for j in range(4):
   with np.load(d1/f'root_observation_{j}.npz')as x,np.load(d/f'root_observation_{j}.npz')as y:
    public_equal &= set(x.files)==set(y.files)and all(np.array_equal(x[k],y[k])for k in x.files)
 assert public_equal,'Same-root public windows differ'
 assert len(rows)==6 and all(r['physics_guard_passed']for r in results)
 return dict(episode_id=rows[0]['episode_id'],root_id=root['root_id'],family_id=root['family_id'],split=root['split'],environment=root['environment'],
  state_kind=root['state_kind'],branch=branch,branch_mode=root['branch_modes'][branch],setup_valid=intended(root,setup['actual_initial_labels']),
  setup_successes=[r['status']for r in setup['results']],initial_categorical=setup['actual_initial_labels']['categorical'],all_physics_guard=True,
  same_root_public_equal=bool(public_equal),heldout_combo_seen=combo,
  quarantine_family=(combo and root['split']in ['train','development']),
  statuses=[r['status']for r in results],official_successes=[r['goal_success']for r in results],
  control_steps=sum(r['control_samples_including_start']-1 for r in results),
  setup_control_steps=len(setup['controls']))
def qa(manifest,audits):
 allowed={'rgb_static','rgb_gripper','depth_static','depth_gripper','robot_obs','seg_static','seg_gripper'}
 for cam in ['static','gripper']:
  allowed|={f'calib_{cam}_{k}'for k in ['width','height','fov','nearval','farval','viewMatrix','projectionMatrix']}
 checked=0;chain={}
 for row in manifest['records'][:128]:
  d=Path(row['labels_path']).parent;lab=read(row['labels_path']);controls=read(d/'controls.json');res=read(d/'result_private.json')
  with gzip.open(d/'trace_private.json.gz','rt')as f:trace=json.load(f)
  assert len(trace)==len(controls)==res['control_samples_including_start']-1
  for name in ['initial_public.npz','final_public.npz']+[f'boundary_history_{j}.npz'for j in range(4)]:
   with np.load(d/name)as z:
    assert set(z.files)<=allowed,(name,z.files)
    assert {'rgb_static','rgb_gripper','depth_static','depth_gripper','robot_obs'}<=set(z.files)
    assert z['rgb_static'].ndim==3 and z['rgb_gripper'].ndim==3
    assert all(np.isfinite(z[k]).all()for k in z.files)
  assert all((d/f'{c}.mp4').stat().st_size>500 for c in ['rgb_static','rgb_gripper'])
  if row['episode_id']in chain:assert chain[row['episode_id']]==lab['before'],'Step labels discontinuous'
  chain[row['episode_id']]=lab['after'];checked+=1
 quarantine=sorted({a['family_id']for a in audits if a['quarantine_family']})
 split_families=collections.defaultdict(set)
 for r in read(S/'data/roots_manifest.json'):split_families[r['split']].add(r['family_id'])
 assert all(not(v&w)for i,v in enumerate(split_families.values())for j,w in enumerate(split_families.values())if i<j)
 return dict(utc=now(),requests_checked=checked,public_allowlist_passed=True,continuous_labels_passed=True,video_and_trace_passed=True,
  split_family_disjoint=True,quarantined_families=quarantine,training_updates=0,features_cached=0,
  no_outcome_replacements=True,no_model_scores_used=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['pilot','all'],required=True);args=ap.parse_args()
 guard();verify();budget();lock=(P/'runs/gpu.lock').open();fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for p,h in read(S/'configs/roots_frozen.json')['sha256'].items():assert sha(p)==h
 if args.phase=='all':assert read(S/'results/pilot_gate.json')['passed']
 roots=read(S/'data/roots_manifest.json')
 # Pilot complete first, then original frozen family order; all splits remain fixed.
 order=sorted(enumerate(roots),key=lambda ir:(not ir[1]['pilot'],ir[0]))
 manifest={'records':[],'task_calls':0,'schema':'factorial_expansion_v1'};audits=[]
 oracle=hydra.utils.instantiate(OmegaConf.load(P/'vendor/calvin/calvin_models/conf/callbacks/rollout/tasks/new_playtable_tasks.yaml'))
 started=time.time();save(S/f'results/{args.phase}_started.json',dict(utc=now(),pid=os.getpid()))
 try:
  for _,root in order:
   if args.phase=='pilot'and not root['pilot']:continue
   for branch in [1,2]:
    budget();d=S/'data/episodes'/f"{root['root_id']}_b{branch}"
    if (d/'COMPLETE.json').exists():rows=read(d/'committed_rows.json')
    else:rows=episode.run_episode(root,branch,oracle)
    a=audit_episode(root,branch,rows);audits.append(a);manifest['records']+=rows;manifest['task_calls']=len(manifest['records'])
    save(S/'data/data_manifest.json',manifest);save(S/'results/episode_audits_private.json',audits)
    quarantine={v['family_id']for v in audits if v['quarantine_family']}
    membership={s:sorted({v['episode_id']for v in audits if v['split']==s and v['family_id']not in quarantine})for s in ['train','development','combination_test']}
    save(S/'data/membership.json',dict(episodes=membership,quarantined_families=sorted(quarantine),no_reassignment=True))
    elapsed=time.time()-read(S/'configs/launch.json')['started_epoch']
    save(S/'results/collection_progress.json',dict(utc=now(),episodes=len(audits),requests=len(manifest['records']),planned_episodes=800,phase=args.phase,last_episode=a['episode_id'],seconds_since_plan=elapsed,split_counts=dict(collections.Counter(v['split']for v in audits)),quarantined_families=len(quarantine)))
    if len(manifest['records'])>=50 and not(S/'results/storage_first50.json').exists():
     n=sum(sum(p.stat().st_size for p in Path(r['labels_path']).parent.rglob('*')if p.is_file())for r in manifest['records'][:50])
     save(S/'results/storage_first50.json',dict(utc=now(),requests=50,bytes=n,bytes_per_request=n/50,projected4800_bytes=n/50*4800,excludes='setup/root snapshots; feature caches deferred'))
    if len(manifest['records'])>=128 and not(S/'results/qa128.json').exists():save(S/'results/qa128.json',qa(manifest,audits))
    print('EPISODE_COMPLETE',len(audits),a['episode_id'],('sealed'if root['split']=='combination_test'else a['statuses']),flush=True)
    if len(audits)%128==0:save(S/f'results/batch_{len(audits):03d}.json',dict(utc=now(),episodes=len(audits),all_physics_guard=True,all_roots_match=True,quarantine_families=sorted(quarantine)))
  if args.phase=='pilot':
   assert len(audits)==32
   q=qa(manifest,audits);valid_kinds=sorted({a['state_kind']for a in audits if a['split']!='combination_test'and a['setup_valid']})
   gate=dict(utc=now(),passed=valid_kinds==[0,1,2,3,4],valid_train_dev_state_kinds=valid_kinds,
     episodes=32,requests=192,qa=q,reason='All five intended state kinds physically reached in pilot'if valid_kinds==[0,1,2,3,4]else'Pilot did not physically reach all five declared initial-state factors; inspect before bulk',
     seconds=time.time()-started,actual_bytes=sum(p.stat().st_size for p in (S/'data').rglob('*')if p.is_file()))
   save(S/'results/pilot_gate.json',gate);print('PILOT_GATE',json.dumps(gate),flush=True)
  else:
   assert len(audits)==800 and len(manifest['records'])==4800
   save(S/'results/collection_complete.json',dict(utc=now(),episodes=800,requests=4800,all_failures_retained=True,replacements=0,training_updates=0,quarantined_families=sorted(quarantine),seconds=time.time()-started))
 except BaseException as e:
  save(S/'results/interruption.json',dict(utc=now(),phase=args.phase,error=repr(e),traceback=traceback.format_exc(),completed_episodes=len(audits),preserved=True))
  raise
if __name__=='__main__':main()
