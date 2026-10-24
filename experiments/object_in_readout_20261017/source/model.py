import torch
from torch import nn
def make():return nn.Sequential(nn.Linear(64,32),nn.SiLU(),nn.Linear(32,1))
def loss(logits,y,v,pos_weight):
 values=torch.nn.functional.binary_cross_entropy_with_logits(logits,y,pos_weight=pos_weight,reduction='none')*v
 return (values.sum(1)/v.sum(1).clamp(min=1)).mean()
def state_hash(q):
 import hashlib
 return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes()for _,v in sorted(q.state_dict().items()))).hexdigest()
