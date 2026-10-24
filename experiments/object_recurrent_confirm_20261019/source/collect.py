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
 guard();verify();budget()
 for p,h in read(S/'configs/roots_frozen.json')['sha256'].items():assert sha(p)==h
 for p,h in read(S/'configs/models_frozen.json')['sha256'].items():assert sha(p)==h
 lock=(P/'runs/gpu.lock').open();fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.monotonic();roots=read(S/'data/roots_manifest.json');manifest={'schema':'object_recurrent_confirm_v1','records':[],'real_transitions':0,'event_names':['red_moved','blue_moved','pink_moved']};audits=[];oracle=hydra.utils.instantiate(OmegaConf.load(P/'vendor/calvin/calvin_models/conf/callbacks/rollout/tasks/new_playtable_tasks.yaml'))
 save(S/'results/collection_started.json',{'utc':now(),'frozen_models_sha256':sha(S/'configs/models_frozen.json'),'frozen_roots_sha256':sha(S/'configs/roots_frozen.json'),'scoring_code_frozen_before_rollouts':True})
 for root in roots:
  for branch in [1,2]:
   budget()
   if time.time()>=read(S/'configs/launch.json')['collection_deadline_epoch']:raise TimeoutError('35-minute collection cutoff, retain partial without comparison')
   eid=f"{root['root_id']}_b{branch}";rows=episode.run_episode(root,branch,oracle);assert len(rows)==6;manifest['records']+=rows;manifest['real_transitions']=len(manifest['records']);save(S/'data/data_manifest.json',manifest);save(S/'results/collection_progress.json',{'utc':now(),'real_transitions':len(manifest['records']),'last_episode':eid,'seconds_this_run':time.monotonic()-start})
   if len(manifest['records'])>=50 and not(S/'results/storage_first50.json').exists():
    n=sum(sum(p.stat().st_size for p in Path(r['obs_before']).parent.rglob('*')if p.is_file())for r in manifest['records'][:50]);save(S/'results/storage_first50.json',{'utc':now(),'bytes_per_transition':n/50,'projected576_bytes':n/50*576,'excludes':'root/setup snapshots and features'})
  audits.append(audit_role(root));save(S/'results/root_audits.json',audits);assert audits[-1]['paired_prefix_exact']and all(x['physics_guard_all']for x in audits[-1]['branches']);print('ROLE_COMPLETE',root['root_id'],'valid_pair',audits[-1]['valid_pair'],flush=True)
  if len(manifest['records'])>=128 and not(S/'results/qa128.json').exists():save(S/'results/qa128.json',{'utc':now(),'real_transitions':len(manifest['records']),'new_geometry_families':len({r['family_id']for r in manifest['records']}),'all_failures_retained':True,'training_updates':0})
 assert len(audits)==48 and len(manifest['records'])==576;save(S/'results/collection_complete.json',{'utc':now(),'seconds':time.monotonic()-start,'total_transitions':576,'role_pairs':48,'valid_pairs':sum(a['valid_pair']for a in audits),'no_replacement':True,'passed':True});print('COLLECTION_COMPLETE',flush=True)
if __name__=='__main__':main()
