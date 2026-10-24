from pathlib import Path
import json,hashlib,datetime
import numpy as np
P=Path('/home/robot/research/projects/wm_symbolic')
E=P/'experiments/visual_predicator_minimal_20261015'
B=P/'data/visual_predicator_test_20261015'
RGB=P/'experiments/state_predicates_rgb_registry_v1_20261014'
STATE=P/'experiments/state_predicates_state_dependence_20261014'
TEMP=P/'experiments/state_predicates_temporal_20261014'
TRAIN=P/'experiments/state_predicates_condition_training_20261015'
CAND=P/'experiments/state_predicates_held_down_20261014/candidate'
COLORS=['red','blue','pink'];FACTS=['Holding(red)','Holding(blue)','Holding(pink)','In(red,drawer)','In(blue,drawer)','In(pink,drawer)','Open(drawer)']
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda a:a.tolist()if isinstance(a,np.ndarray)else a.item()if isinstance(a,np.generic)else str(a))+'\n')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def known(p):
 k=np.isfinite(p);h=p[...,:3]>=.5;k[...,:3]&=~((h.sum(-1)>1)[...,None]&h);return k
def decode(p):return np.where(known(p),p>=.5,-1).astype(int)
def metrics(p,y,v,ids=None):
 if ids is not None:p,y,v=p[ids],y[ids],v[ids]
 k=known(p);h=p>=.5;c=k&(h==y)&v;out={'windows':len(y),'all7':int(c.all(1).sum()),'valid_facts':int(v.sum()),'correct_facts':int(c.sum()),'UNKNOWN':int((v&~k).sum()),'facts':{}}
 for j,n in enumerate(FACTS):
  out['facts'][n]={'positive':int((v[:,j]&y[:,j]).sum()),'negative':int((v[:,j]&~y[:,j]).sum()),'TP':int((v[:,j]&k[:,j]&h[:,j]&y[:,j]).sum()),'TN':int((v[:,j]&k[:,j]&~h[:,j]&~y[:,j]).sum()),'FP':int((v[:,j]&k[:,j]&h[:,j]&~y[:,j]).sum()),'FN':int((v[:,j]&k[:,j]&~h[:,j]&y[:,j]).sum()),'UNKNOWN':int((v[:,j]&~k[:,j]).sum())}
 return out
def normalize(f):
 st=read(CAND/'normalization.json');vm=np.array(st['vision_mean'],np.float32);vs=np.array(st['vision_std'],np.float32);rm=np.array(st['robot_mean'],np.float32);rs=np.array(st['robot_std'],np.float32)
 tok=(f['local_tokens'].astype(np.float32)-vm)/vs;g=f['geometry'];vis=g[:,:,:,0,4].astype(bool);o=np.zeros((4,3,2,390),np.float32)
 for j in range(3):
  if not vis[:,:,j].any():continue
  o[:,j,:,:384]=tok[:,:,j].mean(2)*vis[:,:,j,None];o[:,j,:,384:389]=g[:,:,j,0,:5];o[:,j,:,389]=g[:,:,j,:,5].mean(2)
 grid=(f['global_grid'].astype(np.float32)-vm)/vs
 return {'objects':o.reshape(4,3,780),'context':grid.reshape(4,2,2,2,2,2,384).mean((3,5)).reshape(4,3072),'robot':(f['robot_history']-rm)/rs}
