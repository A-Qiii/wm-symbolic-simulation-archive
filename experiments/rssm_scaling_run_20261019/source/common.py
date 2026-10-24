from pathlib import Path
import json,hashlib,os,datetime,numpy as np
P=Path('/home/robot/research/projects/wm_symbolic')
B=P/'experiments/rssm_scaling_run_20261019'
REPORT=P/'reports/rssm_scaling_run_20261019'
SIZE=int(os.environ.get('RSSM_SCALE_ROOTS','20'));assert SIZE in [20,40]
E=B/'cells'/f'roots{SIZE}'
R=REPORT/f'roots{SIZE}'
DESIGN=P/'experiments/rssm_scaling_design_audit_20261019'
EPREFIX={'vision','geometry','robot_token','spatial','observation'}
ARMS=['joint','route_current_only']
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def default(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 raise TypeError(type(x).__name__)
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.partial');q.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=default)+'\n');q.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def th(state):
 h=hashlib.sha256()
 for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def partition(m):
 e=[p for n,p in m.named_parameters()if n.split('.')[0]in EPREFIX]
 r=[p for n,p in m.named_parameters()if n.split('.')[0]not in EPREFIX]
 assert sum(p.numel()for p in e)==2080768 and sum(p.numel()for p in r)==9432795
 return e,r
def verify():
 for p,h in read(E/'configs/frozen.json')['sha256'].items():assert sha(p)==h,p
def guard(stage):
 import sys,re
 members=read(B/'configs/fixed_design.json')['members']
 allowed=set(members['train'+str(SIZE)]) if stage=='train' else set(members['train20']) if stage=='probe' else set().union(*map(set,members.values()))
 def audit(event,args):
  if event!='open' or not isinstance(args[0],(str,bytes)):return
  p=Path(os.fsdecode(args[0])).resolve();mode=args[1];flags=args[2]if len(args)>2 else 0
  writing=(isinstance(mode,str)and any(c in mode for c in 'wa+'))or(isinstance(flags,int)and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
  if writing and p.is_relative_to(P) and not(p.is_relative_to(B)or p.is_relative_to(REPORT)or p==P/'runs/gpu.lock'):raise RuntimeError('historical write forbidden '+str(p))
  if p.is_relative_to(P/'data/core1200_v1'):
   found=re.search(r'([ABCD]_core1200_[0-9]{2})',str(p))
   if found and found.group(1)not in allowed:raise RuntimeError('Root access forbidden at '+stage+': '+str(p))
  if stage in ['train','probe']and p.is_relative_to(E/'data'):
   if stage=='train':raise RuntimeError('Training must use only prefiltered raw train cache, not derived eval arrays')
   if stage=='probe'and p.suffix=='.npz'and not p.name.endswith('_common20.npz'):raise RuntimeError('Probe forbidden cache '+str(p))
 sys.addaudithook(audit)
