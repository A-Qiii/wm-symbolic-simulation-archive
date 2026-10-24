from pathlib import Path
import json,hashlib,datetime
import numpy as np
P=Path('/home/robot/research/projects/wm_symbolic')
REP=P/'experiments/interaction_latent_repair_20261016'
OLD=P/'experiments/interaction_latent_query_20261016'
PRIOR=P/'experiments/intermediate_data_prediction_20261016'
FIX=P/'experiments/empty_open_readout_correction_20261016'
E=P/'experiments/multistep_chain_source_20261016'
R=P/'reports/multistep_chain_source_20261016'
PACK=PRIOR/'data/transition_train'
COLORS=['red','blue','pink']
FACTS=['Holding(red)','Holding(blue)','Holding(pink)','In(red,drawer)','In(blue,drawer)','In(pink,drawer)','Closed(drawer)','Open(drawer)','Intermediate(drawer)']
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda a:a.tolist() if isinstance(a,np.ndarray) else a.item() if isinstance(a,np.generic) else str(a))+'\n')
def discrete(p):
    p=np.asarray(p);out=p>=.5;out[...,6:]=False;flat=out.reshape(-1,9);flat[np.arange(len(flat)),6+p.reshape(-1,9)[:,6:].argmax(1)]=True;return out
