from pathlib import Path
import json,hashlib,datetime
import numpy as np
P=Path('/home/robot/research/projects/wm_symbolic');REP=P/'experiments/interaction_latent_repair_20261016';FIX=P/'experiments/empty_open_readout_correction_20261016';ABL=P/'experiments/transition_objective_ablation_20261016';E=P/'experiments/independent_objective_D5_20261016';R=P/'reports/independent_objective_D5_20261016'
COLORS=['red','blue','pink'];FACTS=[f'Holding({c})'for c in COLORS]+[f'In({c},drawer)'for c in COLORS]+['Closed(drawer)','Open(drawer)','Intermediate(drawer)']
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda a:a.tolist()if isinstance(a,np.ndarray)else a.item()if isinstance(a,np.generic)else str(a))+'\n')
def action(skill):
 fam='lift'if skill.startswith('lift_')else skill.split('_')[0];a=np.zeros(8,np.float32);a[['open','lift','place','close'].index(fam)]=1
 if fam=='lift':a[4+COLORS.index(skill.split('_')[1])]=1
 a[7]=1;return a.tolist()
def discrete(p):
 p=np.asarray(p);o=p>=.5;o[...,6:]=False;v=o.reshape(-1,9);v[np.arange(len(v)),6+p.reshape(-1,9)[:,6:].argmax(1)]=True;return o
