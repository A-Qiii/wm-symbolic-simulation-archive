from pathlib import Path
import os,sys,json,hashlib,datetime,time,csv
import numpy as np
P=Path('/home/robot/research/projects/wm_symbolic');S=P/'experiments/posterior_history_diagnostic_20261019';G=P/'experiments/broader_action_replay_20261019';D=P/'experiments/action_consequence_diagnostic_20261019';R=P/'experiments/spatial_coverage_readout_run_20261019';O=P/'experiments/object_identity_readout_20261019';B=P/'experiments/rssm_scaling_run_20261019';H=P/'experiments/rssm_h2_supplement_20261019';CELL=B/'cells/roots40';COLORS=['red','blue','pink'];SPLITS=['old_band','gap'];SKILLS=['open_drawer','close_drawer','lift_red_block_table','lift_blue_block_table','lift_pink_block_table','place_in_drawer'];STATUS=['success','refused','executed_failure'];FIELDS=['holding_identity','drawer_membership','drawer_state','all_relations']
os.environ['MPLCONFIGDIR']=str(S/'cache/matplotlib');os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def plain(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {k:plain(v)for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [plain(v)for v in x]
 return x
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.partial');q.write_text(json.dumps(plain(x),ensure_ascii=False,indent=2)+'\n');q.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def csvwrite(p,rows):
 if not rows:return
 with Path(p).open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def guard(event,args):
 if event=='socket.connect':raise RuntimeError('Offline diagnosis')
 if event=='open'and isinstance(args[0],(str,bytes)):
  p=Path(os.fsdecode(args[0])).resolve();m=args[1];f=args[2]if len(args)>2 else 0
  if p.is_relative_to(P):
   if any(t in str(p)for t in ['externalC','C_core1200','D_core1200','independent_identity_D','independent_objective_D','calvin_scene_C','calvin_scene_D','calvin_table_C','calvin_table_D']):raise RuntimeError('C/D data forbidden: '+str(p))
   if ((isinstance(m,str)and any(c in m for c in 'wa+'))or(isinstance(f,int)and f&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)))and not p.is_relative_to(S):raise RuntimeError('Old data are read-only: '+str(p))
def dataset(split):return dict(np.load(S/f'data/dataset_{split}.npz'))
def rel_from_cat(cat):
 c=np.asarray(cat);return np.concatenate([(c[...,:3]==6).astype(float),(c[...,:3]==3).astype(float),np.eye(3)[c[...,3]]],-1)
def rel_from_prob(p):
 p=np.asarray(p);return np.concatenate([p[...,np.array([6,14,22])],p[...,np.array([3,11,19])],p[...,24:27]],-1)
def hard(p):return np.concatenate([(np.asarray(p)[...,:6]>=.5).astype(np.float32),np.eye(3,dtype=np.float32)[np.asarray(p)[...,6:9].argmax(-1)]],-1)
def correct(p,y):
 z=hard(p);held=(z[...,:3]==y[...,:3]).all(-1);inside=(z[...,3:6]==y[...,3:6]).all(-1);door=z[...,6:9].argmax(-1)==y[...,6:9].argmax(-1);return np.stack([held,inside,door,held&inside&door],-1)
def visual_path(seed,split):return (H/'data'if split=='test'else CELL/'data')/f'predictions_joint_seed{seed}_{split}.npz'
def verify_inputs():
 q=read(S/'configs/inputs_frozen.json')['sha256']
 for p,h in q.items():assert sha(p)==h,p
 return len(q)
