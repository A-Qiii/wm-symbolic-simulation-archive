from pathlib import Path
import json,hashlib,datetime,numpy as np
P=Path('/home/robot/research/projects/wm_symbolic');EX=P/'experiments';REP=EX/'interaction_latent_repair_20261016';PRIOR=EX/'intermediate_data_prediction_20261016';FIX=EX/'empty_open_readout_correction_20261016';ABL=EX/'transition_objective_ablation_20261016';WHOLE=EX/'joint_coverage_query_control_20261017';OBJ=EX/'object_in_readout_20261017';COMBO=EX/'combination_generalization_20261017/formal_run_v2';D=EX/'independent_identity_D12_20261017';E=EX/'holding_readout_diagnostic_20261018';R=P/'reports/holding_readout_diagnostic_20261018';COLORS=['red','blue','pink'];METHODS=['actual','one_step','recursive'];FORMS=['object','context']
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda a:a.tolist()if isinstance(a,np.ndarray)else a.item()if isinstance(a,np.generic)else str(a))+'\n')
def features(z,form):
 if form=='object':return z[:,:3]
 return np.concatenate([z[:,:3],np.repeat(z[:,3:4],3,axis=1),np.repeat(z[:,:3].mean(1,keepdims=True),3,axis=1)],axis=-1)
def verify():
 for p,h in read(E/'configs/frozen.json')['sha256'].items():assert sha(p)==h,p
def action(skill,hold=1):
 family='lift'if skill.startswith('lift')else skill.split('_')[0];a=np.zeros(8,np.float32);a[['open','lift','place','close'].index(family)]=1
 if family=='lift':a[4+COLORS.index(skill.split('_')[1])]=1
 a[7]=hold;return a
