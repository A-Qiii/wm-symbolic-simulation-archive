from common import *
import argparse,fcntl,signal
sys.path.insert(0,str(P/'scripts'))
import episode
from omegaconf import OmegaConf
import hydra
def status(l):return 'success'if l['success']else'executed_failure'if l['control_samples_including_start']>1 else'refused'
def root_audit(root):
 rr=[]
 for b in range(3):
  d=S/'data/episodes'/f"{root['root_id']}_b{b}";rows=read(d/'committed_rows.json');ls=[read(r['labels_path'])for r in rows];held=COLORS.index(root['held_color']);target=COLORS.index(root['primary_color']if b==0 else root['matched_color']);fault=ls[1];after=fault['after'];first=ls[0]['after'];setup=read(d/'setup_private.json')['actual_initial_labels']
  rr.append({'branch':b,'episode_id':f"{root['root_id']}_b{b}",'h0_valid':setup['categorical'][held]==6 and setup['categorical'][3]==2,'first_placed_InDrawer':first['categorical'][held]==3,'grasp_status':status(fault),'grasp_controls':fault['control_samples_including_start']-1,'target_held':after['categorical'][target]==6,'empty_hand':after['hand_empty_derived']is True,'retained_InDrawer':after['categorical'][held]==3,'valid_observed_failure':fault['control_samples_including_start']>1 and after['hand_empty_derived']is True and after['categorical'][target]!=6 and first['categorical'][held]==3 and after['categorical'][held]==3,'nominal_grasp_success':fault['success']and after['categorical'][target]==6 and after['categorical'][held]==3,'later_control_counts':[l['control_samples_including_start']-1 for l in ls[2:]],'later_statuses':[status(l)for l in ls[2:]],'all_statuses':[status(l)for l in ls],'gripper_overrides':read(Path(rows[1]['labels_path']).parent/'result_private.json')['requested_applied_gripper_changes'],'physics_guard_all':all(read(Path(r['labels_path']).parent/'result_private.json')['physics_guard_passed']for r in rows)})
 # Compare true public prefixes before the hidden disturbance, not just IDs.
 a=S/'data/episodes'/f"{root['root_id']}_b1";b=S/'data/episodes'/f"{root['root_id']}_b2";files=[('root_observation_'+str(t)+'.npz')for t in range(4)]+[f"00_place_in_drawer/boundary_history_{t}.npz"for t in range(4)]
 exact=True;max_robot=0.
 for name in files:
  x=dict(np.load(a/name));y=dict(np.load(b/name));assert set(x)==set(y);exact&=all(np.array_equal(x[k],y[k])for k in x);max_robot=max(max_robot,float(abs(x['robot_obs']-y['robot_obs']).max()))
 return {'root_id':root['root_id'],'family_id':root['family_id'],'split':root['split'],'environment':root['environment'],'held_color':root['held_color'],'matched_color':root['matched_color'],'branches':rr,'paired_prefix_public_arrays_exact':bool(exact),'prefix_robot_max_difference':max_robot,'valid_matched_root':all(x['h0_valid']for x in rr)and rr[1]['nominal_grasp_success']and rr[2]['valid_observed_failure']and rr[2]['gripper_overrides']>0}
def main():
 pa=argparse.ArgumentParser();pa.add_argument('--phase',choices=['train_dev','evaluation'],required=True);pa.add_argument('--pilot',action='store_true');args=pa.parse_args();guard('collect_train'if args.phase=='train_dev'else'collect_eval');verify_freeze()
 for p,h in read(S/'configs/roots_frozen.json')['sha256'].items():assert sha(p)==h
 if args.phase=='evaluation':
  selected=read(S/'configs/models_frozen.json')
  for p,h in selected['sha256'].items():assert sha(p)==h
  save(S/'results/evaluation_collection_started.json',{'utc':now(),'selection_sha256':sha(S/'configs/models_frozen.json'),'before_evaluation_physical_outcomes':True})
 else:
  if not args.pilot:
   p=read(S/'results/pilot.json');assert p['passed'],p
 lock=(P/'runs/gpu.lock').open('r');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.monotonic();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in()).throw(TimeoutError('1800second collection invocation cap')));signal.alarm(1800)
 roots=[r for r in read(S/'data/roots_manifest.json')if(r['split']=='evaluation')==(args.phase=='evaluation')]
 if args.pilot:roots=[r for r in roots if r['split']=='train'][:6]
 manifest=read(S/'data/data_manifest.json')if(S/'data/data_manifest.json').exists()else{'schema':'failure_recovery_v1','records':[],'real_transitions':0,'event_names':['red_moved','blue_moved','pink_moved']};done={r['episode_id']for r in manifest['records']};audits=read(S/'results/root_audits.json')if(S/'results/root_audits.json').exists()else[]
 oracle=hydra.utils.instantiate(OmegaConf.load(P/'vendor/calvin/calvin_models/conf/callbacks/rollout/tasks/new_playtable_tasks.yaml'))
 for root in roots:
  for b in range(3):
   eid=f"{root['root_id']}_b{b}"
   if eid in done:continue
   rows=episode.run_episode(root,b,oracle);assert len(rows)==6;manifest['records']+=rows;manifest['real_transitions']=len(manifest['records']);save(S/'data/data_manifest.json',manifest)
   save(S/'results/collection_progress.json',{'utc':now(),'phase':args.phase,'real_transitions':len(manifest['records']),'last_episode':eid,'seconds_this_run':time.monotonic()-start})
   if len(manifest['records'])>=50 and not(S/'results/storage_first50.json').exists():
    sizes=[sum(p.stat().st_size for p in Path(r['obs_before']).parent.rglob('*')if p.is_file())for r in manifest['records'][:50]];save(S/'results/storage_first50.json',{'utc':now(),'first50_bytes':sum(sizes),'bytes_per_transition':sum(sizes)/50,'projected432_bytes':sum(sizes)/50*432,'excludes':'setup/root snapshots and DINO cache'})
  if not any(a['root_id']==root['root_id']for a in audits):audits.append(root_audit(root));save(S/'results/root_audits.json',audits)
  if args.phase!='evaluation':
   a=audits[-1];print('ROOT_AUDIT',root['root_id'],'valid_matched',a['valid_matched_root'],'prefix_exact',a['paired_prefix_public_arrays_exact'],flush=True)
  else:print('ROOT_AUDIT',root['root_id'],'sealed',flush=True)
  if len(manifest['records'])>=128 and not(S/'results/qa128.json').exists():
   first=manifest['records'][:128];parts=collections.defaultdict(set)
   for r in manifest['records']:parts[r['root_id']].add(r['split'])
   assert all(len(v)==1 for v in parts.values());save(S/'results/qa128.json',{'utc':now(),'real_transitions_checked':128,'root_leakage':False,'physics_guard_all':all(read(Path(r['labels_path']).parent/'result_private.json')['physics_guard_passed']for r in first),'failures_retained':sum(not read(r['labels_path'])['success']for r in first),'training_not_started':not(S/'configs/models_frozen.json').exists()})
 if args.pilot:
  aa=[a for a in audits if a['root_id']in {r['root_id']for r in roots}];passed=all(any(a['valid_matched_root']and a['environment']==e for a in aa)for e in 'AB')and all(a['paired_prefix_public_arrays_exact']for a in aa);save(S/'results/pilot.json',{'utc':now(),'passed':passed,'roots':len(aa),'valid_roots':sum(a['valid_matched_root']for a in aa),'no_root_replacement':True,'seconds':time.monotonic()-start});print('PILOT_COMPLETE',passed,flush=True)
 else:
  aa=[a for a in audits if a['split']=='train'];valid=sum(a['valid_matched_root']for a in aa);passed=valid>=8 and all(a['paired_prefix_public_arrays_exact']for a in audits);save(S/f'results/collection_{args.phase}_complete.json',{'utc':now(),'seconds':time.monotonic()-start,'all_transitions':len(manifest['records']),'train_valid_matched_roots':valid,'training_gate_passed':passed,'no_root_replacement':True});print('COLLECTION_COMPLETE',args.phase,len(manifest['records']),passed,flush=True)
 signal.alarm(0)
if __name__=='__main__':main()
