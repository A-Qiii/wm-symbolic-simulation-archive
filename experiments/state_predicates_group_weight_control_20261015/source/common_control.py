from pathlib import Path
import json,hashlib,datetime,sys,csv
import numpy as np
P=Path('/home/robot/research/projects/wm_symbolic');E=P/'experiments/state_predicates_group_weight_control_20261015';PREV=P/'experiments/state_predicates_coverage_exploratory5_20261015';DIAG=P/'experiments/state_predicates_fit_diagnostic_20261015';OLD=P/'experiments/state_predicates_one_step_20261014';ABC=P/'experiments/state_predicates_state_dependence_20261014';CB=P/'data/predicate_drawer_factor_C_20261015'
COLORS=['red','blue','pink'];NAMES=['held(red)','held(blue)','held(pink)','in(red,drawer)','in(blue,drawer)','in(pink,drawer)','open(drawer)'];GROUPS={'held':[0,1,2],'in':[3,4,5],'open':[6]}
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
def known(p):
 k=np.isfinite(p);pos=p[:,:3]>=.5;k[:,:3]&=~((pos.sum(1)>1)[:,None]&pos);return k
def csvsave(p,rows):
 if not rows:return
 with Path(p).open('w',newline='',encoding='utf-8-sig')as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows({k:json.dumps(v,ensure_ascii=False)if isinstance(v,(dict,list))else v for k,v in r.items()}for r in rows)
def verify_prior():
 expected=read(DIAG/'results/immutable_after.json')['manifests']+[{'root':str(DIAG),'manifest_sha256':'c9c544aab02ebdbb452a83d1f7621c9c9d5eda5d4d9c8f03a623741e4bc4c21f'}];out=[]
 for old in expected:
  root=Path(old['root']);manifest=root/'delivery_manifest.json';assert sha(manifest)==old['manifest_sha256'];m=read(manifest);entries=m.get('entries',m.get('files'))
  for r in entries:
   p=root/r['path'];assert sha(p)==r['sha256']and not(p.stat().st_mode&0o222),p
  assert len([p for p in root.rglob('*')if p.is_file()])==len(entries)+1
  out.append({'root':str(root),'manifest_sha256':sha(manifest),'entries':len(entries),'all_read_only':True})
 return out
