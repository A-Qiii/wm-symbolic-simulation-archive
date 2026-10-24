from pathlib import Path
import json,hashlib,datetime,numpy as np
P=Path('/home/robot/research/projects/wm_symbolic');EX=P/'experiments';REP=EX/'interaction_latent_repair_20261016';COMBO=EX/'combination_generalization_20261017/formal_run_v2';EV=COMBO.parent/'evaluator_review_v1';OBJ=EX/'object_in_readout_20261017';WHOLE=EX/'joint_coverage_query_control_20261017';E=EX/'independent_identity_D12_20261017';R=P/'reports/independent_identity_D12_20261017';COLORS=['red','blue','pink'];BRANCHES=['normal','reverse_then_lift','place_then_other_lift'];METHODS=['actual','one_step','recursive'];ARMS=['original','object_control','object_coverage'];FACTS=[f'Holding({c})'for c in COLORS]+[f'In({c},drawer)'for c in COLORS]+['Closed','Open','Intermediate']
V1=COMBO.parent/'formal_run_v1'
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
 p=np.asarray(p);o=p>=.5;o[...,6:]=False;q=o.reshape(-1,9);q[np.arange(len(q)),6+p.reshape(-1,9)[:,6:].argmax(1)]=True;return o
def verify_frozen():
 for p,h in read(E/'configs/before_collection_frozen.json')['sha256'].items():assert sha(p)==h,p
