import torch,json
from pathlib import Path
from wm_symbolic.symbol_decoder import Calibration,decode,probabilities,event_probabilities
p={'objects':torch.tensor([[.15,.25,.30,.10,.05,.05,.05,.05],[.8,.1,.02,.02,.02,.02,.01,.01],[.8,.1,.02,.02,.02,.02,.01,.01]]),
   'drawer':torch.tensor([.34,.33,.33]),'slider':torch.tensor([.8,.1,.1]),'lights':torch.tensor([.2,.8])}
names=['red_drawer','red_held','hand_empty','drawer_closed','slider_right','red_moved']
facts,error=decode(p,torch.zeros(7),names,Calibration())
assert facts['red_drawer'] is None and facts['red_held'] is None and facts['drawer_closed'] is None and facts['hand_empty'] is None
assert facts['slider_right'] is True and facts['red_moved'] is False and error=='unknown'
p['objects'][0]=torch.tensor([.8,.1,.02,.02,.02,.02,.01,.01]);p['drawer']=torch.tensor([.8,.1,.1])
facts,error=decode(p,torch.zeros(7),names,Calibration());assert error is None and facts['hand_empty'] is True and facts['red_drawer'] is False
p['objects'][0]=torch.tensor([0.,0.,0.,0.,0.,0.,1.,0.]);p['objects'][1]=p['objects'][0].clone()
assert decode(p,torch.zeros(7),names,Calibration())[1]=='multiple_held_conflict'
assert torch.equal(event_probabilities(torch.zeros(7),Calibration(event_temperature=2)),torch.full((7,),.5))
result={'passed':True,'low_max_categorical_confidence_yields_UNKNOWN_not_all_false':True,'HandEmpty_derived_without_extra_head':True,'multiple_held_rejected':True,'event_temperature_preserves_neutral_probability':True}
Path('/home/robot/research/projects/wm_symbolic/runs/symbol_calibration_regression.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
