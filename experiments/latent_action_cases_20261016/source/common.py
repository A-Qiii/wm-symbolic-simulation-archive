from pathlib import Path
import json,hashlib,datetime
import numpy as np
P=Path('/home/robot/research/projects/wm_symbolic')
OLD=P/'experiments/interaction_latent_query_20261016'
REPAIR=P/'experiments/interaction_latent_repair_20261016'
PREV=P/'experiments/latent_tsne_20261016'
E=P/'experiments/latent_action_cases_20261016'
R=P/'reports/latent_action_cases_20261016'
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda v:v.tolist()if isinstance(v,np.ndarray)else v.item()if isinstance(v,np.generic)else str(v))+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
FACTS=['持红','持蓝','持粉','红块入屉','蓝块入屉','粉块入屉','抽屉关闭','抽屉打开','抽屉中间']
def discrete(p):
 a=np.array(p)>=.5;a[6:]=False;a[6+int(np.argmax(p[6:]))]=True;return a
def semantics(y):
 held='、'.join(c for c,b in zip(['红','蓝','粉'],y[:3])if b)or'空手';inside='、'.join(c for c,b in zip(['红','蓝','粉'],y[3:6])if b)or'无';drawer=['关闭','打开','中间'][int(np.argmax(y[6:]))];return f'持物：{held}；入屉：{inside}；抽屉：{drawer}'
