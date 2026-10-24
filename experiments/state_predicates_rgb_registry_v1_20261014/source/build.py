from pathlib import Path
import json,hashlib,argparse
import numpy as np
from registry import extract,RULE
P=Path('/home/robot/research/projects/wm_symbolic');E=P/'experiments/state_predicates_rgb_registry_v1_20261014';OLD=P/'experiments/state_predicates_long_pairs_20261014'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['train','dev'],required=True);args=ap.parse_args();scope=read(E/'configs/scope.json');assert sha(E/'source/registry.py')==scope['frozen_sha256'][str(E/'source/registry.py')]
 if args.split=='dev':assert read(E/'configs/frozen_before_C.json')['AB_registration_gate_passed']
 assert not(E/'data'/f'windows_{args.split}.json').exists();meta=[r for r in read(OLD/'data/public/windows_full.json')if r['split']==args.split];out=[]
 for r in meta:
  paths=[];region={};info=[]
  for fi,src in zip(r['history_frame_indices'],r['public_frames']):
   dest=E/'data/rgb'/r['root_id']/f'{fi:05d}.npz';dest.parent.mkdir(exist_ok=True)
   # Only these three members are opened from the original archive.
   with np.load(src)as z:public={k:z[k].copy()for k in ['rgb_static','rgb_gripper','robot_obs']}
   assert not dest.exists();np.savez_compressed(dest,**public);paths.append(str(dest));per={}
   for cam in ['static','gripper']:
    masks,detail=extract(public['rgb_'+cam]);region.setdefault(cam,[]).append(masks);per[cam]=detail
   info.append(per)
  rp=E/'data/regions'/(r['window_id']+'.npz');assert not rp.exists();np.savez_compressed(rp,**{k:np.stack(v)for k,v in region.items()});entry={k:v for k,v in r.items()if k not in ['input_path','input_sha256','raw_feature_path','raw_feature_sha256','public_frames']};entry.update(public_frames=paths,source_public_frames=r['public_frames'],sanitized_sha256=[sha(p)for p in paths],rgb_regions_path=str(rp),rgb_regions_sha256=sha(rp),region_details=info);out.append(entry)
 save(E/'data'/f'windows_{args.split}.json',out);print(json.dumps({'split':args.split,'windows':len(out),'sanitized_RGB_proprio_files':4*len(out),'simulator_mask_access_in_builder':False,'future_frame_access':False}),flush=True)
if __name__=='__main__':main()
