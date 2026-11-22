from pathlib import Path
import json,time,hashlib,fcntl,os
# Data, checkpoints and outputs are supplied separately from this core snapshot.
ROOT=Path(os.environ.get('WM_EXPERIMENT_ROOT',Path(__file__).resolve().parent)).expanduser().resolve()
NAMES=['large','small','small2']
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');q.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def status(stage,**kw):
 x=dict(stage=stage,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw);save(ROOT/'STATUS.json',x);print(json.dumps(x,ensure_ascii=False),flush=True)
def plan():return read(ROOT/'configs/plan.json')
def lock():
 p=Path(os.environ.get('WM_GPU_LOCK',str(ROOT/'runs/gpu.lock')));p.parent.mkdir(parents=True,exist_ok=True)
 f=p.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);return f
def episode_path(e):return ROOT/e.get('storage','data/episodes')/e['id']/'episode.json'
