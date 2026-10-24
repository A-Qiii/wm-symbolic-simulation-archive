from common import *
import argparse,fcntl
sys.path.insert(0,str(P/'scripts'))
import episode
from omegaconf import OmegaConf
import hydra
def audit_role(root):
 rows=[]
 for branch in [1,2]:
  d=S/'data/episodes'/f"{root['root_id']}_b{branch}";rr=read(d/'committed_rows.json');ls=[read(r['labels_path'])for r in rr];initial=read(d/'setup_private.json');h=COLORS.index(root['held_color']);t=COLORS.index(root['matched_color'])
  valid=initial['actual_initial_labels']['categorical'][h]==6 and initial['actual_initial_labels']['categorical'][3]==2
  fault=ls[1];after=fault['after'];first=ls[0]['after']
  rows.append({'branch':branch,'initial_valid':bool(valid),'setup_all_success':all(x['status']=='success'for x in initial['results']),'h0_cat':initial['actual_initial_labels']['categorical'],'initial_positions':{c:initial['actual_initial_labels']['continuous']['objects'][c]['position_m']for c in COLORS},'first_H_InDrawer':first['categorical'][h]==3,'target_held':after['categorical'][t]==6,'retained_H_InDrawer':after['categorical'][h]==3,'valid_success':fault['success']and after['categorical'][t]==6 and after['categorical'][h]==3,'valid_failure':fault['control_samples_including_start']>1 and after['hand_empty_derived']is True and after['categorical'][t]!=6 and first['categorical'][h]==3 and after['categorical'][h]==3,'controls':[x['control_samples_including_start']-1 for x in ls],'statuses':['success'if x['success']else'executed_failure'if x['control_samples_including_start']>1 else'refused'for x in ls],'physics_guard_all':all(read(Path(r['labels_path']).parent/'result_private.json')['physics_guard_passed']for r in rr)})
 a=S/'data/episodes'/f"{root['root_id']}_b1";b=S/'data/episodes'/f"{root['root_id']}_b2";exact=True
 for name in [f'root_observation_{t}.npz'for t in range(4)]+[f'00_place_in_drawer/boundary_history_{t}.npz'for t in range(4)]:
  with np.load(a/name)as x,np.load(b/name)as y:exact&=set(x.files)==set(y.files)and all(np.array_equal(x[k],y[k])for k in x.files)
 return {'root_id':root['root_id'],'family_id':root['family_id'],'split':root['split'],'environment':root['environment'],'held_color':root['held_color'],'target_color':root['matched_color'],'branches':rows,'paired_prefix_exact':bool(exact),'valid_pair':all(r['initial_valid']for r in rows)and rows[0]['valid_success']and rows[1]['valid_failure']}
def main():
 pa=argparse.ArgumentParser();pa.add_argument('--phase',choices=['train_dev','locked_test'],required=True);pa.add_argument('--pilot',action='store_true');args=pa.parse_args();stage='collect_train'if args.phase=='train_dev'else'collect_test';guard(stage);verify_freeze();budget_check()
 for p,h in read(S/'configs/roots_frozen.json')['sha256'].items():assert sha(p)==h,p
 if args.phase=='locked_test':
  seal=read(S/'configs/models_frozen.json')
  for p,h in seal['sha256'].items():assert sha(p)==h
  save(S/'results/test_collection_started.json',{'utc':now(),'models_sha256':sha(S/'configs/models_frozen.json'),'models_selected_before_any_test_rollout':True})
 elif not args.pilot:assert read(S/'results/pilot.json')['passed']
 lock=(P/'runs/gpu.lock').open();fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.monotonic();roots=[r for r in read(S/'data/roots_manifest.json')if(r['split']=='locked_test')==(args.phase=='locked_test')]
 if args.pilot:roots=[r for r in roots if r['family_id']in ['A_role_train_00','B_role_train_00']]
 manifest=read(S/'data/data_manifest.json')if(S/'data/data_manifest.json').exists()else{'schema':'role_generalization_v1','records':[],'real_transitions':0,'event_names':['red_moved','blue_moved','pink_moved']};done={r['episode_id']for r in manifest['records']};audits=read(S/'results/root_audits.json')if(S/'results/root_audits.json').exists()else[]
 oracle=hydra.utils.instantiate(OmegaConf.load(P/'vendor/calvin/calvin_models/conf/callbacks/rollout/tasks/new_playtable_tasks.yaml'))
 for root in roots:
  for branch in [1,2]:
   budget_check();eid=f"{root['root_id']}_b{branch}"
   if eid in done:continue
   rows=episode.run_episode(root,branch,oracle);assert len(rows)==6;manifest['records']+=rows;manifest['real_transitions']=len(manifest['records']);save(S/'data/data_manifest.json',manifest)
   save(S/'results/collection_progress.json',{'utc':now(),'phase':args.phase,'real_transitions':len(manifest['records']),'last_episode':eid,'seconds_this_run':time.monotonic()-start})
   if len(manifest['records'])>=50 and not(S/'results/storage_first50.json').exists():
    n=sum(sum(p.stat().st_size for p in Path(r['obs_before']).parent.rglob('*')if p.is_file())for r in manifest['records'][:50]);save(S/'results/storage_first50.json',{'utc':now(),'bytes_per_transition':n/50,'projected1512_bytes':n/50*1512,'excludes':'root/setup snapshots and cached features'})
  if not any(a['root_id']==root['root_id']for a in audits):audits.append(audit_role(root));save(S/'results/root_audits.json',audits)
  print('ROLE_COMPLETE',root['root_id'],'sealed'if root['split']=='locked_test'else audits[-1]['valid_pair'],flush=True)
  if len(manifest['records'])>=128 and not(S/'results/qa128.json').exists():
   rr=manifest['records'][:128];splits=collections.defaultdict(set)
   for r in manifest['records']:splits[r['family_id']].add(r['split'])
   assert all(len(x)==1 for x in splits.values());save(S/'results/qa128.json',{'utc':now(),'first128_real':True,'family_split_leakage':False,'physics_guard_all':all(read(Path(r['labels_path']).parent/'result_private.json')['physics_guard_passed']for r in rr),'failures_retained':sum(not read(r['labels_path'])['success']for r in rr),'training_not_started':not(S/'configs/models_frozen.json').exists()})
 if args.pilot:
  aa=[a for a in audits if a['root_id']in {r['root_id']for r in roots}];passed=all(any(a['environment']==e and a['valid_pair']for a in aa)for e in 'AB')and all(a['paired_prefix_exact']for a in aa);save(S/'results/pilot.json',{'utc':now(),'passed':passed,'role_pairs':len(aa),'valid':sum(a['valid_pair']for a in aa),'seconds':time.monotonic()-start,'no_replacement':True});assert passed,'Pilot physical pair failure; retain evidence'
 else:
  aa=[a for a in audits if(a['split']=='locked_test')==(args.phase=='locked_test')];assert all(a['paired_prefix_exact']and all(b['physics_guard_all']for b in a['branches'])for a in aa);save(S/f'results/collection_{args.phase}_complete.json',{'utc':now(),'seconds':time.monotonic()-start,'total_transitions':len(manifest['records']),'role_pairs':len(aa),'valid_pairs':sum(a['valid_pair']for a in aa),'passed':True,'no_replacement':True})
 print('COLLECTION_COMPLETE',args.phase,args.pilot,len(manifest['records']),flush=True)
if __name__=='__main__':main()
