from pathlib import Path
import json,hashlib,datetime,numpy as np
P=Path('/home/robot/research/projects/wm_symbolic');EX=P/'experiments';REP=EX/'interaction_latent_repair_20261016';PROBE=EX/'simultaneous_relation_probe_20261017';COMBO=EX/'combination_generalization_20261017/formal_run_v2';WHOLE=EX/'joint_coverage_query_control_20261017';E=EX/'object_in_readout_20261017';R=P/'reports/object_in_readout_20261017';COLORS=['red','blue','pink'];FACTS=[f'Holding({c})'for c in COLORS]+[f'In({c},drawer)'for c in COLORS]+['Closed','Open','Intermediate'];PATHS=['actual','one_step','recursive'];BRANCHES=['normal','reverse_then_lift','place_then_other_lift'];ARMS=['original','whole_control','whole_coverage','in_control','in_coverage']
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda a:a.tolist()if isinstance(a,np.ndarray)else a.item()if isinstance(a,np.generic)else str(a))+'\n')
def discrete(p):
 p=np.asarray(p);out=p>=.5;out[...,6:]=False;q=out.reshape(-1,9);q[np.arange(len(q)),6+p.reshape(-1,9)[:,6:].argmax(1)]=True;return out
def verify():
 for p,h in read(E/'configs/frozen.json')['sha256'].items():assert sha(p)==h,p
def verify_seal(p):
 d=read(p)
 for f,h in d['sha256'].items():assert sha(f)==h,f
 return len(d['sha256'])
