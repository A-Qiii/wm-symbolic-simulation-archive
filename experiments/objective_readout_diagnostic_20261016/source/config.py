from pathlib import Path
import json,hashlib,datetime
import numpy as np
P=Path('/home/robot/research/projects/wm_symbolic')
REP=P/'experiments/interaction_latent_repair_20261016'
SPAN=P/'experiments/supervision_span_20261016'
DIAG=P/'experiments/fit_generalization_diagnostic_20261016'
E=P/'experiments/objective_readout_diagnostic_20261016'
R=P/'reports/objective_readout_diagnostic_20261016'
COLORS=['red','blue','pink'];FACTS=[f'Holding({c})'for c in COLORS]+[f'In({c},drawer)'for c in COLORS]+['Closed(drawer)','Open(drawer)','Intermediate(drawer)']
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda a:a.tolist()if isinstance(a,np.ndarray)else a.item()if isinstance(a,np.generic)else str(a))+'\n')
def discrete_logits(x):
 x=np.asarray(x);o=x>=0;o[...,6:]=False;flat=o.reshape(-1,9);flat[np.arange(len(flat)),6+x.reshape(-1,9)[:,6:].argmax(1)]=True;return o
