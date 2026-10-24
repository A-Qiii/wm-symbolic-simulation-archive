from common import *
import argparse,fcntl,time,collections,signal
sys.path.insert(0,str(P/'scripts'))
import core1200_collect as collect
from omegaconf import OmegaConf
import hydra
collect.B=S/'data';collect.R=S/'results'
def h0_audit(root):
 rid=root['root_id'];d=S/'data/episodes'/f'{rid}_b0';actual=read(d/'setup_private.json')['actual_initial_labels'];out={'root_id':rid,'setup':read(d/'setup_private.json'),'h0':actual,'h0_paired_match':None}
 if root['split']=='train':
  old=P/'data/core1200_v1/episodes'/f"{root['paired_old_root']}_b0";label=read(old/'setup_private.json')['actual_initial_labels'];out['old_h0']=label;out['h0_paired_match']=actual['categorical'][:4]==label['categorical'][:4]and actual['categorical_valid'][:4]==label['categorical_valid'][:4]
 # Post-setup actual pose duplicate test, same named colors; no selection by scores.
 cfg=OmegaConf.load(P/f"vendor/calvin/calvin_env/conf/scene/calvin_scene_{root['environment']}.yaml");names=list(cfg.objects.movable_objects)
 with np.load(d/'root_reset_private.npz')as z:a0=z['scene_obs'][6:].reshape(3,6)
 a=np.array([dict(zip(names,a0))['block_'+c]for c in COLORS]);matches=[]
 paths=[Path(x['path'])for x in read(S/'configs/historical_snapshot_inventory.json')if x['environment']==root['environment']]+list((S/'data/roots').glob(f"{root['environment']}_*/branch0_initial_private.npz"))
 for path in paths:
  if path.parent.name==rid:continue
  with np.load(path)as z:b0=z['scene_obs'][6:].reshape(3,6)
  b=np.array([dict(zip(names,b0))['block_'+c]for c in COLORS]);dist=np.linalg.norm(a[:,:3]-b[:,:3],axis=1);yaw=np.abs((a[:,5]-b[:,5]+np.pi/2)%np.pi-np.pi/2)
  if np.all(dist<=.012)or np.all(dist<=.020)and np.all(yaw<=.25):matches.append(str(path))
 out['post_setup_near_duplicates']=matches;return out
def main():
 pa=argparse.ArgumentParser();pa.add_argument('--phase',choices=['train','evaluation'],required=True);pa.add_argument('--max-roots',type=int);args=pa.parse_args();sys.addaudithook(guard)
 frozen=read(S/'configs/roots_frozen.json')
 for p,h in frozen['sha256'].items():assert sha(p)==h
 if args.phase=='evaluation':
  for p,h in read(S/'configs/heads_frozen.json')['sha256'].items():assert sha(p)==h
  save(S/'results/evaluation_collection_started.json',{'utc':now(),'heads_frozen_sha256':sha(S/'configs/heads_frozen.json'),'before_first_formal_eval_trajectory':True})
 assert sha(P/'src/wm_symbolic/physical_oracle.py')==read(S/'configs/protocol.json')['controller_sha256']
 lock=(P/'runs/gpu.lock').open('r');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for sig in [signal.SIGTERM,signal.SIGINT,signal.SIGUSR1]:signal.signal(sig,collect.requested)
 roots=read(S/'data/roots_manifest.json');roots=[r for r in roots if (r['split']=='train')==(args.phase=='train')];roots.sort(key=lambda r:(r['split'],int(r['root_id'][-2:]),r['environment']))
 if args.max_roots:roots=roots[:args.max_roots]
 m=read(S/'data/data_manifest.json')if(S/'data/data_manifest.json').exists()else{'records':[],'real_transitions':0,'event_names':['red_moved','blue_moved','pink_moved']};done={x['episode_id']for x in m['records']};start=time.monotonic();audits=read(S/'results/h0_audit.json')if(S/'results/h0_audit.json').exists()else[]
 oracle=hydra.utils.instantiate(OmegaConf.load(P/'vendor/calvin/calvin_models/conf/callbacks/rollout/tasks/new_playtable_tasks.yaml'))
 for root in roots:
  for b in range(3):
   eid=f"{root['root_id']}_b{b}"
   if eid in done:continue
   if collect.stop:return
   rows=collect.run_episode(root,b,oracle);m['records']+=rows;m['real_transitions']=len(m['records']);save(S/'data/data_manifest.json',m)
   if b==0:
    a=h0_audit(root);audits.append(a);save(S/'results/h0_audit.json',audits)
   save(S/'results/collection_progress.json',{'utc':now(),'phase':args.phase,'transitions':len(m['records']),'split_counts':dict(collections.Counter(x['split']for x in m['records'])),'last_episode':eid,'seconds_this_run':time.monotonic()-start,'pid':os.getpid()})
   if not(S/'results/first_effective_collection.json').exists():save(S/'results/first_effective_collection.json',{'utc':now(),'episode':eid,'real_transitions':len(rows),'no_training_yet':True})
   if len(m['records'])>=50 and not(S/'results/storage_first50.json').exists():
    sizes=[sum(p.stat().st_size for p in Path(r['obs_before']).parent.rglob('*')if p.is_file())for r in m['records'][:50]];save(S/'results/storage_first50.json',{'bytes_per_transition':sum(sizes)/50,'first50_bytes':sum(sizes),'projected720_bytes':sum(sizes)/50*720})
  bad=[a for a in audits if a['h0_paired_match']is False or a['post_setup_near_duplicates']]
  if bad:
   save(S/'results/BLOCKED_COLLECTION.json',{'utc':now(),'reason':'Accepted root h0 pairing or post-setup independence failed; preserve all failures, no replacement','failed_roots':[a['root_id']for a in bad],'transitions':len(m['records']),'training_started':False});print('COLLECTION_GATE_BLOCKED',flush=True);return
  if len(m['records'])>=128 and not(S/'results/qa128.json').exists():
   first=m['records'][:128];assert len({r['root_id']for r in m['records']if r['split']=='train'}&{r['root_id']for r in m['records']if r['split']!='train'})==0
   save(S/'results/qa128.json',{'utc':now(),'transitions':128,'root_leakage':False,'controller_hashes':sorted({r['controller_sha256']for r in first}),'all_physics_guard':all(read(Path(r['labels_path']).parent/'result_private.json')['physics_guard_passed']for r in first),'failed_actions_retained':sum(not read(r['labels_path'])['success']for r in first),'additional_trial_fits':0})
 save(S/f'results/collection_{args.phase}_{args.max_roots or 0}_complete.json',{'utc':now(),'transitions':sum(r['split']=='train'for r in m['records'])if args.phase=='train'else sum(r['split']!='train'for r in m['records']),'seconds':time.monotonic()-start})
if __name__=='__main__':main()
