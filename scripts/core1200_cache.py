from pathlib import Path
import argparse,json,signal,time,os
import numpy as np
from wm_symbolic.vision_cache import FrozenVision,VisibleRegistry,WEIGHTS
P=Path('/home/robot/research/projects/wm_symbolic');B=P/'data/core1200_v1';R=P/'runs/core1200_v1'
stop=False
def request(sig,frame):
 global stop
 stop=True
def main():
 a=argparse.ArgumentParser();a.add_argument('--splits',nargs='+',default=['train','dev']);args=a.parse_args()
 for sig in [signal.SIGUSR1,signal.SIGTERM,signal.SIGINT]:signal.signal(sig,request)
 m=json.loads((B/'data_manifest.json').read_text());by={}
 for r in m['records']:
  if r['split'] in args.splits:by.setdefault(r['episode_id'],[]).append(r)
 out=B/'features';out.mkdir(exist_ok=True);enc=FrozenVision(P);items=[];start=time.monotonic()
 existing=json.loads((out/'index.json').read_text())['items'] if (out/'index.json').exists() else []
 items=[x for x in existing if x['split'] not in args.splits]
 for eid,rows in by.items():
  rows.sort(key=lambda x:x['step']);ed=Path(rows[0]['obs_before']).parent.parent;reg=VisibleRegistry()
  boundaries=[[ed/f'root_observation_{j}.npz' for j in range(4)]]+[[Path(r['obs_after']).parent/f'boundary_history_{j}.npz' for j in range(4)] for r in rows]
  for step,paths in enumerate(boundaries):
   dest=out/f'{eid}_{step}.npz'
   if dest.exists():
    fs=[]
    for path in paths:
     with np.load(path) as z:fs.append(dict(z))
    reg.update(fs)
   else:
    tmp=dest.with_suffix('.tmp.npz');np.savez(tmp,**enc.encode(paths,reg));tmp.replace(dest)
   items.append({'episode_id':eid,'root_id':rows[0]['root_id'],'split':rows[0]['split'],'step':step,'path':str(dest),'public_inputs':[str(x) for x in paths]})
  index={'items':items,'backbone':str(WEIGHTS),'frozen':True,'only_public_inputs':True,'model_inputs_do_not_include_registry_ids':True,'protocol':'existing frozen DINOv2 4-frame dual-view visible-mask ROI+coarse tokens'}
  tmp=out/'index.tmp.json';tmp.write_text(json.dumps(index,indent=2));tmp.replace(out/'index.json')
  (R/'cache_progress.json').write_text(json.dumps({'pid':os.getpid(),'cached_boundaries':len(items),'last_episode':eid,'elapsed_seconds':time.monotonic()-start,'complete':False},indent=2))
  print('CACHED',eid,len(items),flush=True)
  if stop:break
 (R/'cache_progress.json').write_text(json.dumps({'cached_boundaries':len(items),'elapsed_seconds':time.monotonic()-start,'complete':not stop},indent=2))
if __name__=='__main__':main()
