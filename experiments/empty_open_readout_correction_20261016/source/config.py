from pathlib import Path
import json,hashlib,datetime
import numpy as np
P=Path('/home/robot/research/projects/wm_symbolic')
REP=P/'experiments/interaction_latent_repair_20261016'
OLD=P/'experiments/interaction_latent_query_20261016'
PREV=P/'experiments/prediction_query_diagnostics_20261016'
PRIOR=P/'experiments/intermediate_data_prediction_20261016'
E=P/'experiments/empty_open_readout_correction_20261016'
R=P/'reports/empty_open_readout_correction_20261016'
COLORS=['red','blue','pink']
FACTS=['Holding(red)','Holding(blue)','Holding(pink)','In(red,drawer)','In(blue,drawer)','In(pink,drawer)','Closed(drawer)','Open(drawer)','Intermediate(drawer)']
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda a:a.tolist()if isinstance(a,np.ndarray)else a.item()if isinstance(a,np.generic)else str(a))+'\n')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def action(skill):
 family='lift'if skill.startswith('lift_')else skill.split('_')[0];a=np.zeros(8,np.float32);a[['open','lift','place','close'].index(family)]=1
 if family=='lift':a[4+COLORS.index(skill.split('_')[1])]=1
 a[7]=1;return a.tolist()
def discrete(p):
 p=np.asarray(p);out=p>=.5;out[...,6:]=False;flat=out.reshape(-1,9);flat[np.arange(len(flat)),6+p.reshape(-1,9)[:,6:].argmax(1)]=True;return out
