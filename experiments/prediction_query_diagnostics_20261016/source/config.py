from pathlib import Path
import json,hashlib,datetime
import numpy as np
P=Path('/home/robot/research/projects/wm_symbolic')
REP=P/'experiments/interaction_latent_repair_20261016'
OLD=P/'experiments/interaction_latent_query_20261016'
HOR=P/'experiments/recursive_horizon_diagnostic_20261016'
E=P/'experiments/prediction_query_diagnostics_20261016'
R=P/'reports/prediction_query_diagnostics_20261016'
FACTS=['持红','持蓝','持粉','红块入屉','蓝块入屉','粉块入屉','抽屉关闭','抽屉打开','抽屉中间']
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda a:a.tolist()if isinstance(a,np.ndarray)else a.item()if isinstance(a,np.generic)else str(a))+'\n')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def content_hash(path):
 with np.load(path)as z:return hashlib.sha256(b''.join(z[k].tobytes()for k in sorted(z.files))).hexdigest()
def discrete(p):
 p=np.asarray(p);a=p>=.5;a[...,6:]=False;a.reshape(-1,9)[np.arange(a.reshape(-1,9).shape[0]),6+p.reshape(-1,9)[:,6:].argmax(1)]=True;return a
