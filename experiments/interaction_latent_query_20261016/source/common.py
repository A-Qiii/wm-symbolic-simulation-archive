from pathlib import Path
import json,hashlib,datetime
import numpy as np
P=Path('/home/robot/research/projects/wm_symbolic')
E=P/'experiments/interaction_latent_query_20261016'
R=P/'reports/interaction_latent_query_20261016'
BASE=P/'experiments/visual_state_semantics_v2_20261015'
COLORS=['red','blue','pink']
FACTS=[f'Holding({c})'for c in COLORS]+[f'In({c},drawer)'for c in COLORS]+['Closed(drawer)','Open(drawer)','Intermediate(drawer)']
DINO=Path('/home/robot/research/models/facebook--dinov2-small/ed25f3a31f01632728cabb09d1542f84ab7b0056')
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda a:a.tolist()if isinstance(a,np.ndarray)else a.item()if isinstance(a,np.generic)else str(a))+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def family(skill):return 'lift'if skill.startswith('lift_')else skill.split('_')[0]
def action(skill,target=None,hold=96):
 typ=['open','lift','place','close'].index(family(skill));v=np.zeros(8,np.float32);v[typ]=1
 if family(skill)=='lift':target=COLORS.index(skill.split('_')[1])
 if target is not None and family(skill)in['lift','place']:v[4+int(target)]=1
 v[7]=hold/96
 return v.tolist()
