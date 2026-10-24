from experiment import *
import numpy as np,sys
GROUPS={'held':[0,1,2],'in':[3,4,5],'open':[6]}
C=P/'experiments/state_predicates_drawer_factor_C_20261015';CB=P/'data/predicate_drawer_factor_C_20261015'
def known(p):
 k=np.isfinite(p);positive=p[:,:3]>=.5;conflict=positive.sum(1)>1;k[:,:3]&=~(conflict[:,None]&positive);return k
def symbol_inputs(p,a):return np.concatenate([np.where(known(p),p>=.5,-1),a],1).astype(np.float32)
def old_modules():
 sys.path.insert(0,str(OLD/'source'))
 from common import reader,logits,digest_model,balanced_bce
 from transition import LatentTransition
 return reader,logits,digest_model,balanced_bce,LatentTransition
def dataset_frozen():
 q=frozen();d=read(E/'configs/dataset_frozen.json')
 for p,h in d['files_sha256'].items():assert sha(p)==h,p
 return q,d
def csvsave(path,rows):
 import csv
 if not rows:return
 with Path(path).open('w',newline='',encoding='utf-8-sig')as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows({k:json.dumps(v,ensure_ascii=False)if isinstance(v,(dict,list))else v for k,v in r.items()}for r in rows)
