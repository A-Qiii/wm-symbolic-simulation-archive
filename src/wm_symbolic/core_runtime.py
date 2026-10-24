"""Core model runtime inherits public-only updates and learned-only search."""
from pathlib import Path
import json,hashlib
import torch
from wm_symbolic.model_runtime import ModelRuntime
from wm_symbolic.world_model import WorldModel
from wm_symbolic.vision_cache import FrozenVision,VisibleRegistry
from wm_symbolic.symbol_decoder import Calibration
from wm_symbolic.search import LearnedSearch

class CoreRuntime(ModelRuntime):
 def __init__(self,project,checkpoint,calibration_path):
  self.project=Path(project);self.checkpoint=Path(checkpoint);ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
  assert ck['identity']['mode']=='core1200_main' and ck['identity']['random_initialization'] and not ck['identity']['old_checkpoint_loaded']
  assert all(x.startswith(('A_core1200_','B_core1200_')) for x in ck['identity']['train_roots'])
  self.model=WorldModel(skill_count=6,event_count=3).eval();self.model.load_state_dict(ck['model']);self.stats=ck['normalization'];self.dynamics=self.model;self.symbol_checkpoint=None;self.method='core_RSSM_LTLf'
  self.vision=FrozenVision(self.project,device='cpu',reserve_gpu=False);self.registry=VisibleRegistry();self.state=None;self.last_prior=None;self.last_before=None;self.last_action=None;self.last_event_logits=None
  names=json.loads((self.project/'runs/core1200_v1/protocol.json').read_text())['skills'];self.catalog=[{'skill':n,'family':'core'} for n in names];self.budgets={'core':900}
  cal=json.loads(Path(calibration_path).read_text());assert cal['checkpoint_sha256']==hashlib.sha256(self.checkpoint.read_bytes()).hexdigest();self.calibration=Calibration(**cal['calibration'])
  self.searcher=LearnedSearch(self.model,self.catalog,self.budgets,self.calibration)
