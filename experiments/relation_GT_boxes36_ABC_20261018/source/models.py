import torch,hashlib
from torch import nn
def make(arm):
 dims={'object':(128,67),'pair':(261,33),'geometry':(5,1244)};i,h=dims[arm];return nn.Sequential(nn.Linear(i,h),nn.SiLU(),nn.Linear(h,1))
def tensor_hash(state):
 h=hashlib.sha256()
 for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()
