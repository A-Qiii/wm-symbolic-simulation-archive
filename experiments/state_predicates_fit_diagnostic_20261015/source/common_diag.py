from pathlib import Path
import json,hashlib,datetime,sys
import numpy as np
P=Path('/home/robot/research/projects/wm_symbolic')
E=P/'experiments/state_predicates_fit_diagnostic_20261015'
PREV=P/'experiments/state_predicates_coverage_exploratory5_20261015'
OLD=P/'experiments/state_predicates_one_step_20261014'
B=P/'data/predicate_condition_coverage_control_20261015'
NAMES=['持红','持蓝','持粉','红入屉','蓝入屉','粉入屉','屉开'];COLORS=['red','blue','pink']
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for c in iter(lambda:f.read(2**20),b''):h.update(c)
 return h.hexdigest()
def old_modules():
 sys.path.insert(0,str(OLD/'source'))
 from common import reader,logits,digest_model,balanced_bce
 from transition import LatentTransition
 return reader,logits,digest_model,balanced_bce,LatentTransition
def frozen():
 q=read(E/'configs/protocol.json')
 for p,h in q['frozen_sha256'].items():assert sha(p)==h,p
 return q
def data():
 import torch
 q=frozen();ids=q['joint_global_ids'];z=dict(np.load(PREV/'data/training/public/inputs.npz'));t=dict(np.load(PREV/'data/training/_private/targets.npz'))
 d={k:torch.tensor(z[k][ids],device='cuda')for k in ['current_objects','current_scene','declared_actions']}
 d.update({k:torch.tensor(t[k][ids],device='cuda')for k in ['next_objects','next_scene','after_y','after_valid']});return q,d,z,t
def correctness(prob,y):
 p=np.asarray(prob);yy=np.asarray(y);k=np.isfinite(p);pos=p[:,:3]>=.5;k[:,:3]&=~((pos.sum(1)>1)[:,None]&pos)
 ok=k&((p>=.5)==yy);return {'all7':int(ok.all(1).sum()),'calls':len(p),'fact_correct':ok.sum(0).tolist(),'UNKNOWN_facts':int((~k).sum()),'predictions':np.where(k,(p>=.5).astype(int),-1).tolist()}
def verify_prior():
 out=[]
 for manifest in sorted((P/'experiments').glob('*/delivery_manifest.json')):
  # Only relevant immutable experimental evidence, not unrelated projects.
  if manifest.parent.name not in ['state_predicates_independent_D_20261015','state_predicates_condition_training_20261015','state_predicates_drawer_factor_C_20261015','state_predicates_condition_coverage_control_20261015','state_predicates_coverage_exploratory5_20261015','state_predicates_implementation_diagnostic_20261015']:continue
  out.append(verify_one(manifest))
 for n in ['predicate_independent_D_20261015','predicate_drawer_factor_C_20261015','predicate_condition_coverage_control_20261015']:out.append(verify_one(P/'data'/n/'delivery_manifest.json'))
 return out
def verify_one(manifest):
 m=read(manifest);entries=m.get('entries',m.get('files'));root=manifest.parent
 for e in entries:
  p=root/e['path'];assert p.is_file()and sha(p)==e['sha256']and not(p.stat().st_mode&0o222),p
 assert len([p for p in root.rglob('*')if p.is_file()])==len(entries)+1
 return {'root':str(root),'manifest_sha256':sha(manifest),'entries':len(entries),'all_read_only':True}
