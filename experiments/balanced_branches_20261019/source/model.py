"""Single change: actual observation can update the persistent deterministic state."""
import torch
from torch import nn
import torch.nn.functional as F
from wm_symbolic.world_model import WorldModel as OriginalWorldModel,State

class ObservationStateWorldModel(OriginalWorldModel):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.observation_to_h=nn.Linear(512,512,bias=False)
        nn.init.zeros_(self.observation_to_h.weight)

    def observe(self,predicted,observation,deterministic=False):
        original,q=super().observe(predicted,observation,deterministic)
        innovation=self.observation_to_h(F.layer_norm(observation,(512,)))
        return State(predicted.h+innovation.to(predicted.h.dtype),original.z),q

def shared_parameters(model):
    return [p for n,p in model.named_parameters() if not n.startswith('observation_to_h.')]

def load_shared(model,state):
    result=model.load_state_dict(state,strict=False)
    assert result.missing_keys==['observation_to_h.weight'] and not result.unexpected_keys
    assert torch.count_nonzero(model.observation_to_h.weight)==0

def add_new_optimizer_group(optimizer,model):
    optimizer.add_param_group({'params':list(model.observation_to_h.parameters()),'lr':.0001,'betas':(.9,.999),'weight_decay':.0001})
