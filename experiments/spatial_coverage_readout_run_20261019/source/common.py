from pathlib import Path
import os,sys,json,hashlib,datetime
import numpy as np
P=Path('/home/robot/research/projects/wm_symbolic')
S=P/'experiments/spatial_coverage_readout_run_20261019'
O=P/'experiments/object_identity_readout_20261019'
B=P/'experiments/rssm_scaling_run_20261019'
Q=P/'experiments/spatial_coverage_readout_plan_20261019'
COLORS=['red','blue','pink']
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def plain(v):
 if isinstance(v,np.ndarray):return v.tolist()
 if isinstance(v,np.generic):return v.item()
 if isinstance(v,dict):return {k:plain(x)for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [plain(x)for x in v]
 return v
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.partial');q.write_text(json.dumps(plain(x),ensure_ascii=False,indent=2)+'\n');q.replace(p)
def guard(event,args):
 if event=='socket.connect':raise RuntimeError('No network in this experiment')
 if event=='open'and isinstance(args[0],(str,bytes)):
  p=Path(os.fsdecode(args[0])).resolve();m=args[1];f=args[2]if len(args)>2 else 0
  writing=(isinstance(m,str)and any(x in m for x in 'wa+'))or(isinstance(f,int)and f&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
  if writing and p.is_relative_to(P)and not p.is_relative_to(S):raise RuntimeError('Old project write forbidden: '+str(p))
def seals():
 out=read(Q/'protected_before.json')
 for name,h in read(Q/'manifest.json')['sha256'].items():out[str(Q/name)]=h
 out[str(Q/'manifest.json')]=sha(Q/'manifest.json')
 for f,h in out.items():assert sha(f)==h,f
 return out
def cls(x):return 0 if x<-.14666666666666667 else 1 if x<.14666666666666672 else 2
