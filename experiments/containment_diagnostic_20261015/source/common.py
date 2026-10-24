from pathlib import Path
import json,hashlib,datetime
import numpy as np
P=Path('/home/robot/research/projects/wm_symbolic');BASE=P/'experiments/visual_state_semantics_v2_20261015';BASE_B=P/'data/visual_state_semantics_v2_ABC_20261015';E=P/'experiments/containment_diagnostic_20261015';B=P/'data/containment_diagnostic_ABC_20261015';R=P/'reports/containment_diagnostic_20261015';OLD=P/'experiments/visual_predicator_minimal_20261015';TRANSFER=P/'experiments/visual_predicator_transfer_query_20261015';CAND=P/'experiments/state_predicates_held_down_20261014/candidate';RGB=P/'experiments/state_predicates_rgb_registry_v1_20261014'
COLORS=['red','blue','pink'];FACTS=['Holding(red)','Holding(blue)','Holding(pink)','In(red,drawer)','In(blue,drawer)','In(pink,drawer)','Closed(drawer)','Open(drawer)','Intermediate(drawer)']
def read(p):return json.loads(Path(p).read_text())
def save(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda a:a.tolist()if isinstance(a,np.ndarray)else a.item()if isinstance(a,np.generic)else str(a))+'\n')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
