from pathlib import Path
import json,hashlib,datetime,numpy as np
P=Path('/home/robot/research/projects/wm_symbolic')
BASE=P/'experiments/endpoint_box_resolution_224_448_20261018'
NAME='endpoint_box_decoder_all_candidates_20261018'
E=P/'experiments'/NAME
R=P/'reports'/NAME
CATS=['drawer_front','gripper_finger']
SPLITS=['train','AB_episode_heldout_development','C_seen_diagnostic','D_seen_diagnostic']
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=lambda x:x.tolist()if isinstance(x,np.ndarray)else x.item()if isinstance(x,np.generic)else str(x))+'\n')
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
def iou(a,b):
 a=np.asarray(a);b=np.asarray(b);inter=np.maximum(np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]),0).prod();u=np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-inter
 return float(inter/max(float(u),1e-9))
