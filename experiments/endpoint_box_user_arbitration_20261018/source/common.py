from pathlib import Path
import json,hashlib,datetime,numpy as np
P=Path('/home/robot/research/projects/wm_symbolic')
BASE=P/'experiments/endpoint_box_localizer_20261018'
REVIEW=Path('/home/robot/Documents/Codex/2026-10-18/task/review')
V1=P/'experiments/endpoint_box_label_adjudication8_20261018'
NAME='endpoint_box_user_arbitration_20261018';E=P/'experiments'/NAME;R=P/'reports'/NAME
IDS=[0,1,8,9,32,33,38,39];CATS=['drawer_front','gripper_finger']
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda a:a.tolist()if isinstance(a,np.ndarray)else a.item()if isinstance(a,np.generic)else str(a))+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def check_manifest(p):
 for f,h in read(p)['sha256'].items():assert sha(f)==h,f
def label_guard(event,args):
 if event=='open'and isinstance(args[0],(str,bytes)):
  p=Path(args[0]).resolve();s=str(p)
  if p.is_relative_to(P/'experiments')and any(q in s for q in ['/results/','/models/','/data/features/','/data/train/','/data/evaluation/']):raise RuntimeError('No model outputs or features during label adjudication')
