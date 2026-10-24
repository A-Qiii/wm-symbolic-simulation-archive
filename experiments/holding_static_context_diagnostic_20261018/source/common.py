from pathlib import Path
import json,hashlib,datetime,os,numpy as np
P=Path('/home/robot/research/projects/wm_symbolic')
B=P/'experiments/relation_GT_boxes36_ABC_20261018'
NAME='holding_static_context_diagnostic_20261018'
E=P/'experiments'/NAME;R=P/'reports'/NAME
ARMS=['context_zero','context_present']
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda v:v.tolist()if isinstance(v,np.ndarray)else v.item()if isinstance(v,np.generic)else str(v))+'\n')
def npz(p,**x):
 with Path(p).open('wb')as f:np.savez_compressed(f,**x)
def make():
 import torch
 return torch.nn.Sequential(torch.nn.Linear(389,33),torch.nn.SiLU(),torch.nn.Linear(33,1))
def tensor_hash(state):
 h=hashlib.sha256()
 for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def masked(x,arm):
 x=x.copy()
 if arm=='context_zero':x[:,261:]=0
 elif arm!='context_present':raise ValueError(arm)
 return x
def guard(stage):
 import sys
 def audit(event,args):
  if event=='open'and isinstance(args[0],(str,bytes)):
   p=Path(os.fsdecode(args[0])).resolve();mode=args[1];flags=args[2]if len(args)>2 else 0
   writing=isinstance(mode,str)and any(c in mode for c in 'wa+')or isinstance(flags,int)and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)
   if writing and (p.is_relative_to(P/'experiments')or p.is_relative_to(P/'reports'))and not(p.is_relative_to(E)or p.is_relative_to(R)):raise RuntimeError('Frozen artifacts write forbidden '+str(p))
   if p.is_relative_to(P/'experiments')and not(p.is_relative_to(E)or p.is_relative_to(B)):raise RuntimeError('Unrelated experiment read forbidden '+str(p))
   if stage in ['features','train','predict']and(p.is_relative_to(B/'data/private')or p==B/'data/records.json'):raise RuntimeError('Private annotations forbidden '+str(p))
   if stage=='train'and(p==B/'data/supervision/dev_targets.json'or p==B/'data/features/dev.npz'):raise RuntimeError('Dev input forbidden during fitting')
   if stage in ['features','predict']and p.is_relative_to(B/'data/supervision'):raise RuntimeError('Labels forbidden during prediction')
 sys.addaudithook(audit)

DINO=P.parent.parent/'models/facebook--dinov2-small/ed25f3a31f01632728cabb09d1542f84ab7b0056'
