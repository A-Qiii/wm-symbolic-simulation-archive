from pathlib import Path
import os,sys,json,hashlib,datetime,time,csv,collections
import numpy as np
P=Path('/home/robot/research/projects/wm_symbolic')
S=P/'experiments/factorial_expansion_20261020'
COLORS=['red','blue','pink']
os.environ['MPLCONFIGDIR']=str(S/'cache/matplotlib')
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
sys.path.insert(0,str(P/'src'))
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def plain(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {str(k):plain(v)for k,v in x.items()}
 if isinstance(x,(tuple,list)):return [plain(v)for v in x]
 return x
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.partial')
 tmp.write_text(json.dumps(plain(x),ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def budget():
 if (S/'STOP').exists():raise RuntimeError('STOP requested; preserve partial data')
 if time.time()>=read(S/'configs/launch.json')['deadline_epoch']:raise TimeoutError('Six-hour collection budget reached; preserve partial data')
 if __import__('shutil').disk_usage(S).free<30*1024**3:raise RuntimeError('Free disk below30GiB')
def guard():
 def audit(event,args):
  if event=='socket.connect':raise RuntimeError('Offline experiment')
  if event!='open'or not isinstance(args[0],(str,bytes)):return
  p=Path(os.fsdecode(args[0])).resolve();m=args[1];f=args[2]if len(args)>2 else 0
  if not p.is_relative_to(P):return
  writing=(isinstance(m,str)and any(c in m for c in 'wa+'))or(isinstance(f,int)and f&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
  if writing and not p.is_relative_to(S):raise RuntimeError('Historical write forbidden '+str(p))
  if any(t in str(p)for t in ['externalC','C_core1200','D_core1200','independent_identity_D','independent_objective_D','calvin_scene_C','calvin_scene_D']):raise RuntimeError('C/D forbidden')
 sys.addaudithook(audit)
def verify():
 for p,h in read(S/'configs/design_frozen.json')['sha256'].items():assert sha(p)==h,p
