from pathlib import Path
import json,torch
from wm_symbolic.symbol_transition import SymbolTransition
from wm_symbolic.world_model import State
torch.set_num_threads(2);torch.manual_seed(0);model=SymbolTransition().eval()
state=State(torch.randn(8,32),torch.zeros(8,1,1));action=model.action(torch.zeros(8,dtype=torch.long),torch.zeros(8,3,dtype=torch.long),torch.full((8,),120.))
with torch.no_grad():
 first,_=model.imagine(state,action);changed_z,_=model.imagine(State(state.h,torch.randn_like(state.z)*1e8),action)
 assert torch.equal(first.h,changed_z.h),'Placeholder state leaked into dynamics'
 shifted=state.h.clone()
 for start,end in [(0,8),(8,16),(16,24),(24,27),(27,30)]:shifted[:,start:end]+=3
 changed_shift,_=model.imagine(State(shifted,state.z),action);error=float((first.h-changed_shift.h).abs().max());assert error<1e-6
 try:model.imagine(State(torch.randn(8,512),torch.zeros(8,32,32)),action)
 except ValueError:pass
 else:raise AssertionError('Continuous RSSM latent was accepted by symbolic baseline')
result={'passed':True,'dynamic_input':'32 symbolic probability values plus known action only','continuous_RSSM_state_rejected':True,
        'unused_state_placeholder_poison_has_zero_effect':True,'categorical_logit_offset_invariance_max_error':error,
        'scope':'Baseline interface unit test only; baseline is not trained'}
Path('/home/robot/research/projects/wm_symbolic/runs/symbol_transition_interface.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
