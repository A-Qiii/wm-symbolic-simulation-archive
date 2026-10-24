from pathlib import Path
import json,hashlib,datetime,numpy as np
P=Path('/home/robot/research/projects/wm_symbolic');EX=P/'experiments';COMBO=EX/'combination_generalization_20261017/formal_run_v2';EV=COMBO.parent/'evaluator_review_v1';REP=EX/'interaction_latent_repair_20261016';PRIOR=EX/'intermediate_data_prediction_20261016';FIX=EX/'empty_open_readout_correction_20261016';E=EX/'simultaneous_relation_probe_20261017';R=P/'reports/simultaneous_relation_probe_20261017';COLORS=['red','blue','pink']
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda a:a.tolist()if isinstance(a,np.ndarray)else a.item()if isinstance(a,np.generic)else str(a))+'\n')
def verify():
 for p,h in read(E/'configs/frozen.json')['sha256'].items():assert sha(p)==h,p
