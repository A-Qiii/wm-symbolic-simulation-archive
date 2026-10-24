import torch
from torch import nn
def make(form):return nn.Sequential(nn.Linear(64 if form=='object'else 192,32),nn.SiLU(),nn.Linear(32,1))
def loss(q,x,y,v,weight):
 l=torch.nn.functional.binary_cross_entropy_with_logits(q(x).squeeze(-1),y,pos_weight=weight,reduction='none')*v
 return (l.sum(1)/v.sum(1).clamp(min=1)).mean()
