from pathlib import Path
import json,hashlib,datetime
import numpy as np
P=Path('/home/robot/research/projects/wm_symbolic');OLD=P/'experiments/interaction_latent_query_20261016';REPAIR=P/'experiments/interaction_latent_repair_20261016';E=P/'experiments/latent_tsne_20261016';R=P/'reports/latent_tsne_20261016'
COLORS=['red','blue','pink']
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda v:v.tolist()if isinstance(v,np.ndarray)else v.item()if isinstance(v,np.generic)else str(v))+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
BASES={'ABC':OLD/'data','D':OLD/'data/D','D2':REPAIR/'data/D2'}
