from pathlib import Path
import json, hashlib, datetime, numpy as np
P=Path('/home/robot/research/projects/wm_symbolic')
BASE=P/'experiments/endpoint_box_localizer_20261018'
NAME='endpoint_box_resolution_224_448_20261018'
E=P/'experiments'/NAME
R=P/'reports'/NAME
DINO=Path('/home/robot/research/models/facebook--dinov2-small/ed25f3a31f01632728cabb09d1542f84ab7b0056')
CATS=['drawer_front','gripper_finger']
SPLITS=['train','AB_episode_heldout_development','C_seen_diagnostic','D_seen_diagnostic']
def read(p): return json.loads(Path(p).read_text())
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=lambda x:x.tolist()if isinstance(x,np.ndarray)else x.item()if isinstance(x,np.generic)else str(x))+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def check_manifest(p):
 entries=read(p)['sha256']
 for f,h in entries.items():assert sha(f)==h,f
 return len(entries)
