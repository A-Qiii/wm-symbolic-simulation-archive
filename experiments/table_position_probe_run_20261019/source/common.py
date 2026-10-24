from pathlib import Path
import os,sys,json,hashlib,datetime,csv,time
import numpy as np
P=Path('/home/robot/research/projects/wm_symbolic');S=P/'experiments/table_position_probe_run_20261019';R=P/'experiments/spatial_coverage_readout_run_20261019';O=P/'experiments/object_identity_readout_20261019';Q=P/'experiments/table_position_probe_plan_20261019';B=P/'experiments/rssm_scaling_run_20261019'
SPLITS=['train','validation','old_band','gap'];COLORS=['red','blue','pink'];CUTS=np.array([-.14666666666666667,.14666666666666672]);LO=-.44;WIDTH=.88
os.environ['MPLCONFIGDIR']=str(S/'cache/matplotlib');os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['TRITON_CACHE_DIR']=str(S/'cache/triton')
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
 if event=='socket.connect':raise RuntimeError('Offline experiment')
 if event=='open'and isinstance(args[0],(str,bytes)):
  p=Path(os.fsdecode(args[0])).resolve();m=args[1];flags=args[2]if len(args)>2 else 0
  if p.is_relative_to(P):
   if any(t in str(p)for t in ['externalC','C_core1200','D_core1200','calvin_scene_C','calvin_scene_D','calvin_table_C','calvin_table_D']):raise RuntimeError('C/D sample access forbidden: '+str(p))
   writing=(isinstance(m,str)and any(c in m for c in 'wa+'))or(isinstance(flags,int)and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
   if writing and not p.is_relative_to(S):raise RuntimeError('Sealed input write forbidden: '+str(p))
def source(split):return O if split=='validation'else R
def features(seed,split):return dict(np.load(source(split)/f'data/features_seed{seed}_{split}.npz'))
def target(split):return dict(np.load(S/f'data/targets_{split}.npz'))
def classes(x):return (np.asarray(x)[...,None]>=CUTS).sum(-1)
def pack(raw):
 n=len(raw['global']);return np.concatenate([raw['object'],np.broadcast_to(raw['global'][:,None],(n,3,512)),np.broadcast_to(np.eye(3),(n,3,3)),raw['availability'][...,None]],-1).astype(np.float32)
def root_values(values,mask,roots):
 out=[];ids=[]
 for r in sorted(set(roots)):
  m=mask&(roots[:,None]==r)
  if m.any():out.append(float(np.asarray(values)[m].mean()));ids.append(str(r))
 return np.asarray(out),ids
def verify_inputs():
 entries=read(S/'configs/inputs_frozen.json')['sha256']
 for p,h in entries.items():assert sha(p)==h,p
 return len(entries)
