from pathlib import Path
import json,hashlib,datetime
import numpy as np
P=Path('/home/robot/research/projects/wm_symbolic');EX=P/'experiments';BASE=EX/'visual_node_edge_readout_20261018'
E=EX/'visual_edge_zero_context_control_20261018';R=P/'reports/visual_edge_zero_context_control_20261018'
HEADS=['Holding','In'];COLORS=['red','blue','pink'];BOX=np.array([.10,.52,.98,1.00])
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=lambda x:x.tolist()if isinstance(x,np.ndarray)else x.item()if isinstance(x,np.generic)else str(x))+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def make():
 from torch import nn
 return nn.Sequential(nn.Linear(207,32),nn.SiLU(),nn.Linear(32,1))
def tensor_hash(s):return hashlib.sha256(b''.join(k.encode()+v.detach().cpu().contiguous().numpy().tobytes()for k,v in sorted(s.items()))).hexdigest()

