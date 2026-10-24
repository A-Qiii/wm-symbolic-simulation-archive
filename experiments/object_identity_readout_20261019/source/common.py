from pathlib import Path
import os,sys,json,hashlib,datetime
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
P=Path('/home/robot/research/projects/wm_symbolic')
B=P/'experiments/rssm_scaling_run_20261019'
S=P/'experiments/object_identity_readout_20261019'
os.environ['MPLCONFIGDIR']=str(S/'cache/matplotlib')
os.environ['TRITON_CACHE_DIR']=str(S/'cache/triton')
import numpy as np
import torch
from torch import nn
COLORS=['red','blue','pink']
SPLITS={'train':'train40','validation':'validation10','test':'test10','externalC':'C20'}
ARMS=['global','object']
UPDATES=1500
BATCH=64
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def default(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 raise TypeError(type(x).__name__)
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.partial');q.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=default)+'\n');q.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for q in iter(lambda:f.read(4*1024**2),b''):h.update(q)
 return h.hexdigest()
def th(state):
 h=hashlib.sha256()
 for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def protected_seals():
 files={}
 manifests=[B/'configs/delivery_manifest.json',P/'experiments/rssm_h2_supplement_20261019/manifest.json']
 for mpath in manifests:
  m=read(mpath)
  for name,h in m['sha256'].items():
   p=Path(name) if Path(name).is_absolute() else mpath.parent/name
   assert sha(p)==h,str(p);files[str(p)]=h
  files[str(mpath)]=sha(mpath)
 return {'files':len(files),'manifest_hashes':{str(p):sha(p)for p in manifests},'sha256':files}
class Head(nn.Module):
 def __init__(self):
  super().__init__();self.identity=nn.Embedding(3,32)
  self.objects=nn.Sequential(nn.LayerNorm(1057),nn.Linear(1057,256),nn.SiLU(),nn.Linear(256,8))
  self.drawer=nn.Sequential(nn.LayerNorm(512),nn.Linear(512,64),nn.SiLU(),nn.Linear(64,3))
 def forward(self,focal,context,availability,identities=None):
  if identities is None:identities=torch.arange(3,device=focal.device).expand(focal.shape[0],3)
  x=torch.cat([focal,context[:,None].expand(-1,3,-1),self.identity(identities),availability[...,None]],-1)
  return self.objects(x),self.drawer(context)
def probabilities(out):
 ob,dr=out;return torch.cat([ob.softmax(-1).flatten(1),dr.softmax(-1)],1)
def audit_writes(event,args):
 if event!='open' or not isinstance(args[0],(str,bytes)):return
 p=Path(os.fsdecode(args[0])).resolve();mode=args[1];flags=args[2]if len(args)>2 else 0
 writing=(isinstance(mode,str)and any(c in mode for c in 'wa+'))or(isinstance(flags,int)and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
 if writing and p.is_relative_to(P) and not p.is_relative_to(S):raise RuntimeError('Original project write forbidden: '+str(p))
 if p.is_relative_to(P/'data/core1200_v1') and '/D_core1200_' in str(p):raise RuntimeError('D sample access forbidden')
