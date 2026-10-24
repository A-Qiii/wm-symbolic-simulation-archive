from pathlib import Path
import os,sys,json,hashlib,datetime,time,csv,collections,itertools,re
import numpy as np
P=Path('/home/robot/research/projects/wm_symbolic');S=P/'experiments/failure_recovery_control_20261019'
B=P/'experiments/rssm_scaling_run_20261019';CELL=B/'cells/roots40'
T=P/'experiments/terminal_posterior_supervision_20261019';R=P/'experiments/spatial_coverage_readout_run_20261019'
COLORS=['red','blue','pink'];SKILLS=['open_drawer','close_drawer','lift_red_block_table','lift_blue_block_table','lift_pink_block_table','place_in_drawer'];ARMS=['nominal','recovery']
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['MPLCONFIGDIR']=str(S/'cache/matplotlib');os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def plain(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {str(k):plain(v)for k,v in x.items()}
 if isinstance(x,(tuple,list)):return [plain(v)for v in x]
 return x
def save(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.partial');q.write_text(json.dumps(plain(obj),ensure_ascii=False,indent=2)+'\n');q.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def statehash(m):
 h=hashlib.sha256()
 for k,v in sorted(m.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def guard(stage):
 def audit(event,args):
  if event=='socket.connect':raise RuntimeError('Offline experiment')
  if event!='open'or not isinstance(args[0],(str,bytes)):return
  p=Path(os.fsdecode(args[0])).resolve();m=args[1];f=args[2]if len(args)>2 else 0
  if not p.is_relative_to(P):return
  writing=(isinstance(m,str)and any(c in m for c in 'wa+'))or(isinstance(f,int)and f&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
  if writing and not p.is_relative_to(S):raise RuntimeError('Historical write forbidden '+str(p))
  if any(t in str(p)for t in ['externalC','C_core1200','D_core1200','independent_identity_D','independent_objective_D','calvin_scene_C','calvin_scene_D','calvin_table_C','calvin_table_D']):raise RuntimeError('C/D forbidden '+str(p))
  if stage in ['train','cache_train','collect_train']and ('_recovery_evaluation_'in str(p)or p.name.endswith('_evaluation.npz')):raise RuntimeError('Locked new evaluation forbidden '+str(p))
 sys.addaudithook(audit)
def csvwrite(p,rows):
 if not rows:return
 with Path(p).open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def verify_freeze():
 d=read(S/'configs/design_frozen.json')
 for p,h in d['sha256'].items():assert sha(p)==h,p
 return len(d['sha256'])
def rel(p):
 p=np.asarray(p);return np.concatenate([p[...,np.array([6,14,22])],p[...,np.array([3,11,19])],p[...,24:27]],-1)
def truth(c):
 c=np.asarray(c);return np.concatenate([c[...,:3]==6,c[...,:3]==3,np.eye(3)[c[...,3]]],-1)
def correct(p,y):
 p=np.asarray(p);y=np.asarray(y);h=((p[...,:3]>=.5)==y[...,:3]).all(-1);i=((p[...,3:6]>=.5)==y[...,3:6]).all(-1);d=p[...,6:9].argmax(-1)==y[...,6:9].argmax(-1);return np.stack([h,i,d,h&i&d],-1)
def nll(p,y):
 p=np.clip(np.asarray(p,np.float64),1e-9,1-1e-9);y=np.asarray(y,np.float64);return (-((y[...,:3]*np.log(p[...,:3])+(1-y[...,:3])*np.log1p(-p[...,:3]))).mean(-1)-((y[...,3:6]*np.log(p[...,3:6])+(1-y[...,3:6])*np.log1p(-p[...,3:6]))).mean(-1)-(y[...,6:9]*np.log(p[...,6:9])).sum(-1))/3
def partition(m):
 e=[p for n,p in m.named_parameters()if n.split('.')[0]in {'vision','geometry','robot_token','spatial','observation'}];r=[p for n,p in m.named_parameters()if n.split('.')[0]not in {'vision','geometry','robot_token','spatial','observation'}];return e,r
