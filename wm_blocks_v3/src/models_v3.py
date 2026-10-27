"""One further intervention: learned per-axis, rather than cross-axis influence.
No support labels/rules; coefficient magnitude/sign are learned from experience.
"""
import torch
from torch import nn
from models_v2 import StateActorDynamics
class DiagonalOutput(nn.Linear):
 def forward(self,x):return torch.diag_embed(super().forward(x)).flatten(-2)
class DiagonalStateDynamics(StateActorDynamics):
 def __init__(self):
  super().__init__();old=self.motion.interaction[-1];new=DiagonalOutput(128,3)
  with torch.no_grad():new.weight.copy_(old.weight[[0,4,8]]);new.bias.copy_(old.bias[[0,4,8]])
  self.motion.interaction[-1]=new
