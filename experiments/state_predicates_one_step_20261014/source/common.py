from pathlib import Path
import json,hashlib,datetime,sys
import numpy as np
import torch
from torch.nn import functional as F
P=Path('/home/robot/research/projects/wm_symbolic');E=P/'experiments/state_predicates_one_step_20261014';DATA=P/'experiments/state_predicates_rgb_registry_v1_20261014';CAND=P/'experiments/state_predicates_held_down_20261014/candidate';B=P/'data/predicate_long_pairs_20261014';sys.path.insert(0,str(CAND))
from model import PredicateReader
from metrics import known
COLORS=['red','blue','pink'];NAMES=['held(red)','held(blue)','held(pink)','in(red,drawer)','in(blue,drawer)','in(pink,drawer)','open(drawer)'];GROUPS={'held':[0,1,2],'in':[3,4,5],'open':[6]}
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def digest_model(m):
 h=hashlib.sha256()
 for k,v in m.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def reader(device='cuda'):
 m=PredicateReader().to(device);m.load_state_dict(torch.load(CAND/'reader_seed1.pt',map_location=device,weights_only=False)['model']);return m.eval().requires_grad_(False)
def encode(m,objects,context,robot):
 b,t,k,_=objects.shape;obj=m.object_encoder(objects);ctx=m.context_encoder(torch.cat([context,robot],-1));joint=torch.cat([obj,ctx[:,:,None].expand(-1,-1,k,-1)],-1).permute(0,2,1,3).reshape(b*k,t,256);_,h=m.object_history(joint);_,s=m.scene_history(ctx);return h[-1].reshape(b,k,96),s[-1]
def logits(m,h,s):
 o=m.object_head(h);d=m.drawer_head(s);return torch.cat([o[:,:,0],o[:,:,1],d],-1)
def balanced_bce(logit,y,v,weights):
 loss=F.binary_cross_entropy_with_logits(logit,y,reduction='none');parts=[]
 for gi,cols in enumerate(GROUPS.values()):
  yy=y[:,cols];vv=v[:,cols];w=torch.where(yy>.5,weights[gi,1],weights[gi,0]);parts.append((loss[:,cols]*w*vv).sum()/vv.sum().clamp(min=1))
 return torch.stack(parts).mean()
def root_selection_score(logit,y,v,weights,roots):
 return torch.stack([balanced_bce(logit[roots==r],y[roots==r],v[roots==r],weights)for r in roots.unique()]).mean()
