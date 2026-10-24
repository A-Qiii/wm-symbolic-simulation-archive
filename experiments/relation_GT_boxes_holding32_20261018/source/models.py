from common import *
import torch
from torch import nn
def make(arm):return nn.Sequential(nn.Linear(128 if arm=='object'else 261,67 if arm=='object'else 33),nn.SiLU(),nn.Linear(67 if arm=='object'else 33,1))
def tensor_hash(state):return hashlib.sha256(b''.join(k.encode()+v.detach().cpu().numpy().tobytes()for k,v in sorted(state.items()))).hexdigest()
