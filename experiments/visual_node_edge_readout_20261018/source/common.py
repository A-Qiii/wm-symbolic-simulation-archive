from pathlib import Path
import json,hashlib,datetime
import numpy as np
P=Path('/home/robot/research/projects/wm_symbolic');EX=P/'experiments'
E=EX/'visual_node_edge_readout_20261018';R=P/'reports/visual_node_edge_readout_20261018'
OLD=EX/'official_readout_coverage_control_20261018';EXP=EX/'official_AB_state_expansion_20261018'
REP=EX/'interaction_latent_repair_20261016';PRIOR=EX/'intermediate_data_prediction_20261016';D=EX/'independent_identity_D12_20261017'
DINO=Path('/home/robot/research/models/facebook--dinov2-small/ed25f3a31f01632728cabb09d1542f84ab7b0056')
COLORS=['red','blue','pink'];HEADS=['Holding','In'];ARMS=['node','edge']
ROIS={'Holding':{'camera':1,'box':[0.,0.,1.,1.],'meaning':'whole wrist-camera gripper context; fixed camera-role region, not jaw segmentation'},'In':{'camera':0,'box':[.10,.52,.98,1.],'meaning':'lower static-camera drawer region; fixed pixel ROI, not automatic drawer detection'}}
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=lambda x:x.tolist()if isinstance(x,np.ndarray)else x.item()if isinstance(x,np.generic)else str(x))+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def tensor_hash(s):return hashlib.sha256(b''.join(k.encode()+v.detach().cpu().contiguous().numpy().tobytes()for k,v in sorted(s.items()))).hexdigest()
def make(arm):
 from torch import nn
 return nn.Sequential(nn.Linear(74 if arm=='node' else 207,88 if arm=='node' else 32),nn.SiLU(),nn.Linear(88 if arm=='node' else 32,1))

