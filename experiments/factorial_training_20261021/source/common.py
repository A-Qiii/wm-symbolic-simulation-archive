from pathlib import Path
import os,sys,json,hashlib,datetime,time,csv,collections
import numpy as np
P=Path('/home/robot/research/projects/wm_symbolic');S=P/'experiments/factorial_training_20261021';D=P/'experiments/factorial_expansion_20261020';M=P/'experiments/midstart_training_control_20261020';O=P/'experiments/object_recurrent_state_run_20261019';OLD=P/'experiments/rgbd_observation_control_20261020';CELL=P/'experiments/rssm_scaling_run_20261019/cells/roots40'
COLORS=['red','blue','pink'];ARMS=['old_control','expanded'];SEEDS=[0,1];UPDATES=1500
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['MPLCONFIGDIR']=str(S/'cache/matplotlib');os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';sys.path.insert(0,str(P/'src'))
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def plain(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {str(k):plain(v)for k,v in x.items()}
 if isinstance(x,(tuple,list)):return [plain(v)for v in x]
 return x
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.partial');t.write_text(json.dumps(plain(x),ensure_ascii=False,indent=2)+'\n');t.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def statehash(m):
 h=hashlib.sha256()
 for k,v in sorted(m.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def csvwrite(p,rr):
 if not rr:return
 with Path(p).open('w',newline='')as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rr for k in r)));w.writeheader();w.writerows(rr)
def budget(*args):
 if(S/'STOP').exists():raise RuntimeError('STOP file present')
 if __import__('shutil').disk_usage(S).free<30*1024**3:raise RuntimeError('Free space below30GiB')
def verify():
 for p,h in read(S/'configs/design_frozen.json')['sha256'].items():assert sha(p)==h,p
 for seal in ['design_frozen.json','roots_frozen.json']:
  for p,h in read(D/'configs'/seal)['sha256'].items():assert sha(p)==h,p
 assert read(S/'results/full_data_qa.json')['passed']
def guard():
 def audit(event,args):
  if event=='socket.connect':raise RuntimeError('Offline experiment')
  if event!='open'or not isinstance(args[0],(str,bytes)):return
  p=Path(os.fsdecode(args[0])).resolve();m=args[1];f=args[2]if len(args)>2 else 0
  if not p.is_relative_to(P):return
  writing=(isinstance(m,str)and any(c in m for c in 'wa+'))or(isinstance(f,int)and f&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
  if writing and not p.is_relative_to(S):raise RuntimeError('Historical write forbidden '+str(p))
 sys.addaudithook(audit)
def rel(p):return np.concatenate([p[...,np.array([6,14,22])],p[...,np.array([3,11,19])],p[...,24:27]],-1)
def truth(c):return np.concatenate([c[...,:3]==6,c[...,:3]==3,np.eye(3)[c[...,3]]],-1)
def correct(p,y):
 h=((p[...,:3]>=.5)==y[...,:3]).all(-1);i=((p[...,3:6]>=.5)==y[...,3:6]).all(-1);d=p[...,6:9].argmax(-1)==y[...,6:9].argmax(-1);return np.stack([h,i,d,h&i&d],-1)
