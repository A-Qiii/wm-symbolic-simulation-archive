from pathlib import Path
import json,hashlib,datetime,sys,importlib.util
import numpy as np
P=Path('/home/robot/research/projects/wm_symbolic');E=P/'experiments/state_predicates_condition_training_20261015';OLD=P/'experiments/state_predicates_one_step_20261014';NEW=P/'experiments/state_predicates_state_dependence_20261014';COPY=P/'experiments/state_predicates_copy_effect_diagnostic_20261014';CAND=P/'experiments/state_predicates_held_down_20261014/candidate';RGB=P/'experiments/state_predicates_rgb_registry_v1_20261014'
COLORS=['red','blue','pink'];NAMES=['held(red)','held(blue)','held(pink)','in(red,drawer)','in(blue,drawer)','in(pink,drawer)','open(drawer)'];GROUPS={'held':[0,1,2],'in':[3,4,5],'open':[6]}
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def loadmod(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def known(p):
 k=np.isfinite(p);conflict=(p[:,:3]>=.5).sum(1)>1;k[:,:3]&=~(conflict[:,None]&(p[:,:3]>=.5));return k
def symbol_inputs(p,actions):
 k=known(p);return np.concatenate([np.where(k,(p>=.5).astype(np.float32),-1),actions],1).astype(np.float32)
def frozen():
 q=read(E/'configs/protocol.json')
 for p,h in q['frozen_sha256'].items():assert sha(p)==h,p
 return q
def old_modules():
 sys.path.insert(0,str(OLD/'source'))
 from common import reader,logits,digest_model,balanced_bce,root_selection_score
 from transition import LatentTransition
 return reader,logits,digest_model,balanced_bce,root_selection_score,LatentTransition

