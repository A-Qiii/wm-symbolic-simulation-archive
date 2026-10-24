from common import *
import fcntl,hydra
from omegaconf import OmegaConf
sys.path.insert(0,str(P/'scripts'))
import episode
def main():
 guard();verify();budget();lock=(P/'runs/gpu.lock').open();fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for p,h in read(S/'configs/roots_frozen.json')['sha256'].items():assert sha(p)==h
 roots=read(S/'data/roots_manifest.json');manifest={'records':[],'task_calls':0,'schema':'frozen_relation_generalization_v1'};audits=[];started=time.time();oracle=hydra.utils.instantiate(OmegaConf.load(P/'vendor/calvin/calvin_models/conf/callbacks/rollout/tasks/new_playtable_tasks.yaml'))
 save(S/'results/collection_started.json',{'utc':now(),'fixed_models_sha256':sha(S/'configs/models_frozen.json'),'frozen_plan_sha256':sha(S/'configs/protocol.json'),'no_added_controls_for_Held_probes':True})
 for root in roots:
  for branch in [1,2]:
   budget()
   if time.time()>=read(S/'configs/launch.json')['collection_deadline_epoch']:raise TimeoutError('Collection cutoff; retain all partial data')
   rows=episode.run_episode(root,branch,oracle);assert len(rows)==len(root['branches'][branch]);manifest['records']+=rows;manifest['task_calls']=len(manifest['records']);save(S/'data/data_manifest.json',manifest)
   d=S/'data/episodes'/f"{root['root_id']}_b{branch}";labs=[read(r['labels_path'])for r in rows];setup=read(d/'setup_private.json');initial=setup['actual_initial_labels'];h=COLORS.index(root['held_color']);valid=initial['categorical'][3]==2 and(initial['categorical'][h]==6 if root['template']=='recovery_seen6'else initial['categorical'][h]==3 and not any(initial['holding_derived']))
   audit={'episode_id':rows[0]['episode_id'],'root_id':root['root_id'],'family_id':root['family_id'],'template':root['template'],'branch':branch,'setup_valid':bool(valid),'setup_successes':[r['status']for r in setup['results']],'statuses':[read(Path(r['labels_path']).parent/'result_private.json')['status']for r in rows],'probe_windows':sum(len(read(Path(r['labels_path']).parent/'held_probes_private.json')['probes'])for r in rows),'all_physics_guard':all(read(Path(r['labels_path']).parent/'result_private.json')['physics_guard_passed']for r in rows)};assert audit['all_physics_guard'];audits.append(audit);save(S/'results/episode_audits.json',audits)
   if branch==2:
    first=S/'data/episodes'/f"{root['root_id']}_b1";checks=[f'root_observation_{j}.npz'for j in range(4)]+[f"{t:02d}_{root['branches'][1][t]}/boundary_history_{j}.npz"for t in range(root['fault_step'])for j in range(4)];exact=True
    for name in checks:
     with np.load(first/name)as x,np.load(d/name)as y:exact &= set(x.files)==set(y.files)and all(np.array_equal(x[k],y[k])for k in x.files)
    audit['matched_prefix_exact']=bool(exact);save(S/'results/episode_audits.json',audits);assert exact,'Public pre-fault prefix differs'
   save(S/'results/collection_progress.json',{'utc':now(),'episodes':len(audits),'task_calls':len(manifest['records']),'last_episode':rows[0]['episode_id'],'seconds':time.time()-started})
   if len(manifest['records'])>=50 and not(S/'results/storage_first50.json').exists():
    n=sum(sum(p.stat().st_size for p in Path(r['labels_path']).parent.rglob('*')if p.is_file())for r in manifest['records'][:50]);save(S/'results/storage_first50.json',{'utc':now(),'bytes_per_task_call':n/50,'projected336_bytes':n/50*336,'excludes':'setup/root snapshots and feature caches'})
   if len(manifest['records'])>=128 and not(S/'results/qa128.json').exists():save(S/'results/qa128.json',{'utc':now(),'task_calls':len(manifest['records']),'episodes':len(audits),'training_updates':0,'all_failures_retained':True,'model_not_evaluated_during_collection':True})
   print('EPISODE_COMPLETE',len(audits),rows[0]['episode_id'],audit['statuses'],flush=True)
 assert len(audits)==72 and len(manifest['records'])==336
 save(S/'results/collection_complete.json',{'utc':now(),'passed':True,'episodes':72,'task_calls':336,'seconds':time.time()-started,'setup_valid':sum(r['setup_valid']for r in audits),'probe_windows':sum(r['probe_windows']for r in audits),'all_failures_retained':True,'replacements':0})
if __name__=='__main__':main()
