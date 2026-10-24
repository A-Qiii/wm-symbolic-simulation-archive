from pathlib import Path
import json,hashlib,datetime,numpy as np
P=Path('/home/robot/research/projects/wm_symbolic');NAME='endpoint_sync_truth_4state_20261018';E=P/'experiments'/NAME;R=P/'reports'/NAME
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda x:x.tolist()if isinstance(x,np.ndarray)else x.item()if isinstance(x,np.generic)else x.decode()if isinstance(x,bytes)else str(x))+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def mesh(path):
 s=Path(path).read_text().splitlines();v=np.array([[float(x)for x in a.split()[1:]]for a in s if a.startswith('v ')]);f=np.array([[int(x.split('/')[0])-1 for x in a.split()[1:]]for a in s if a.startswith('f ')]);assert f.shape[1]==3;return v,f
def bbox(mask):
 y,x=np.where(mask);return [int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)]if len(x)else None
