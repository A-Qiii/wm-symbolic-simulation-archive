from common import *
import argparse,fcntl
from wm_symbolic.vision_cache import FrozenVision,VisibleRegistry,WEIGHTS
def main():
 guard();budget();verify();assert read(S/'results/collection_complete.json')['passed']
 lock=(P/'runs/gpu.lock').open('r');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.monotonic();encoder=FrozenVision(P,reserve_gpu=False);items=read(S/'data/features/index.json')['items']if(S/'data/features/index.json').exists()else[];existing={(r['episode_id'],r['step'])for r in items};by=collections.defaultdict(list)
 for r in read(S/'data/data_manifest.json')['records']:
  by[r['episode_id']].append(r)
 out=S/'data/features';out.mkdir(exist_ok=True)
 for eid,rr in sorted(by.items()):
  budget()
  rr.sort(key=lambda r:r['step']);d=Path(rr[0]['obs_before']).parent.parent;windows=[[d/f'root_observation_{j}.npz'for j in range(4)]]+[[Path(r['obs_after']).parent/f'boundary_history_{j}.npz'for j in range(4)]for r in rr];registry=VisibleRegistry()
  for t,paths in enumerate(windows):
   target=out/f'{eid}_{t}.npz'
   if(eid,t)in existing:registry.update([dict(np.load(p))for p in paths]);continue
   data=encoder.encode(paths,registry);np.savez(target,**data);items.append({'episode_id':eid,'root_id':rr[0]['root_id'],'split':rr[0]['split'],'step':t,'path':str(target),'public_inputs':[str(p)for p in paths]})
  save(out/'index.json',{'items':items,'only_public_inputs':True,'frozen_DINO':str(WEIGHTS),'disturbance_status_branchIDs_not_model_inputs':True});save(S/'results/cache_progress.json',{'utc':now(),'phase':'confirm','boundaries':len(items),'last_episode':eid,'seconds':time.monotonic()-start});print('CACHE',eid,len(items),flush=True)
 save(S/'results/cache_complete.json',{'utc':now(),'boundaries':len(items),'seconds':time.monotonic()-start,'frozen_weights_sha256':sha(WEIGHTS/'model.safetensors')});print('CACHE_COMPLETE','confirm',len(items),flush=True)
if __name__=='__main__':main()
