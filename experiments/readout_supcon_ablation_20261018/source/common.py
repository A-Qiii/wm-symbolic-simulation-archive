from pathlib import Path
import json,hashlib,datetime
import numpy as np
P=Path('/home/robot/research/projects/wm_symbolic');EX=P/'experiments'
OLD=EX/'official_readout_coverage_control_20261018'
E=EX/'readout_supcon_ablation_20261018';R=P/'reports/readout_supcon_ablation_20261018'
REP=EX/'interaction_latent_repair_20261016';PRIOR=EX/'intermediate_data_prediction_20261016'
COLORS=['red','blue','pink'];HEADS=['Holding','In'];ARMS=['bce','supcon']
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
 return nn.Sequential(nn.Linear(64,32),nn.SiLU(),nn.Linear(32,1))
def tensor_hash(state):return hashlib.sha256(b''.join(k.encode()+v.detach().cpu().contiguous().numpy().tobytes()for k,v in sorted(state.items()))).hexdigest()

