"""Model-side runtime. No reset metadata, scene state, checker, or label imports."""
from pathlib import Path
from contextlib import contextmanager
import json,hashlib
import numpy as np
import torch
from wm_symbolic.world_model import WorldModel,State
from wm_symbolic.vision_cache import FrozenVision,VisibleRegistry
from wm_symbolic.gpu_phase import gpu_phase
from wm_symbolic.dataset import static_arguments
from wm_symbolic.search import LearnedSearch
from wm_symbolic.symbol_decoder import Calibration,probabilities,decode,goal_score

def moved(state,device):return State(state.h.to(device),state.z.to(device))

class ModelRuntime:
 def __init__(self,project,checkpoint,calibration_path=None):
  self.project=Path(project);self.checkpoint=Path(checkpoint);ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
  if any(not root.startswith(('A_train_','B_train_')) for root in ck['identity']['train_roots']):raise ValueError('Checkpoint trained outside A/B roots')
  events=ck['model']['outcome.5.weight'].shape[0]-1;self.model=WorldModel(event_count=events).eval();self.model.load_state_dict(ck['model']);self.stats=ck['normalization']
  self.vision=FrozenVision(self.project,device='cpu',reserve_gpu=False);self.registry=VisibleRegistry();self.state=None;self.last_prior=None;self.last_before=None;self.last_action=None
  self.catalog=json.loads((self.project/'configs/skills.json').read_text());self.budgets=json.loads((self.project/'configs/executor_budgets_pilot.json').read_text())['steps'];self.calibration=Calibration()
  if calibration_path:
   data=json.loads(Path(calibration_path).read_text())
   if data['checkpoint_sha256']!=hashlib.sha256(self.checkpoint.read_bytes()).hexdigest():raise ValueError('Calibration checkpoint mismatch')
   self.calibration=Calibration(**data['calibration'])
  self.searcher=LearnedSearch(self.model,self.catalog,self.budgets,self.calibration)
 @contextmanager
 def active(self,vision=False):
  with gpu_phase(self.project,'WM_observation_and_planning'):
   self.model.cuda()
   if vision:
    self.vision.device='cuda:0';self.vision.model.cuda();self.vision.mean=self.vision.mean.cuda();self.vision.std=self.vision.std.cuda()
   try:yield
   finally:
    self.model.cpu()
    if vision:
     self.vision.device='cpu';self.vision.model.cpu();self.vision.mean=self.vision.mean.cpu();self.vision.std=self.vision.std.cpu()
    torch.cuda.empty_cache()
 def encode_public(self,paths):
  f=self.vision.encode(paths,self.registry);stats=self.stats
  vm=np.asarray(stats['vision_mean'],np.float32);vs=np.asarray(stats['vision_std'],np.float32);rm=np.asarray(stats['robot_mean'],np.float32);rs=np.asarray(stats['robot_std'],np.float32)
  tensors=[torch.from_numpy(x).unsqueeze(0).cuda() for x in [(f['tokens'].astype(np.float32)-vm)/vs,f['geometry'],f['valid'],(f['robot']-rm)/rs]]
  return self.model.encode(*tensors).repeat(8,1)
 @torch.no_grad()
 def update(self,public_frames,executed_skill=None,reobserve=False):
  with self.active(vision=True),torch.autocast('cuda',dtype=torch.bfloat16):
   encoded=self.encode_public(public_frames)
   if self.state is None or reobserve and self.last_prior is None:
    state,_=self.model.initial(encoded);events=torch.zeros(8,self.model.event_count,device='cuda')
   elif reobserve:
    prior=moved(self.last_prior,'cuda');state,_=self.model.observe(prior,encoded);events=self.model.outcome_logits(moved(self.last_before,'cuda'),self.last_action.cuda(),prior).float()[:,1:].sigmoid()
   else:
    if executed_skill is None:raise ValueError('History update needs the actually executed skill')
    idx=next(i for i,s in enumerate(self.catalog) if s['skill']==executed_skill);skill=self.catalog[idx];before=moved(self.state,'cuda')
    action=self.model.action(torch.full((8,),idx,device='cuda',dtype=torch.long),torch.tensor([static_arguments(executed_skill)]*8,device='cuda'),torch.full((8,),self.budgets[skill['family']],device='cuda',dtype=torch.float32))
    prior,_=self.model.imagine(before,action);state,_=self.model.observe(prior,encoded);events=self.model.outcome_logits(before,action,prior).float()[:,1:].sigmoid()
    self.last_before=moved(before,'cpu');self.last_prior=moved(prior,'cpu');self.last_action=action.cpu()
   self.current_probs={k:v.cpu() for k,v in probabilities(self.model.symbols(state),self.calibration).items()};self.event_probs=events.cpu();self.state=moved(state,'cpu')
  return self.belief_record()
 def belief_record(self):
  return {'symbol_probabilities':{k:v.tolist() for k,v in self.current_probs.items()},'event_probabilities':self.event_probs.tolist(),
   'source':'Actual public observation posterior; events estimated by the trained prior-event head for the executed command; not observed ground-truth events','calibrated':self.calibration.calibrated,'ground_truth_read':False}
 def facts(self,names):
  facts,reason=decode({k:v.mean(0) for k,v in self.current_probs.items()},self.event_probs.mean(0),names,self.calibration)
  for j,color in enumerate(['red','blue','pink']):
   if j not in self.registry.colour_ids:
    for name in names:
     if name.startswith(color+'_'):facts[name]=None;reason='object_never_visually_registered'
  return facts,reason
 def current_goal_score(self,task):
  if any(j not in self.registry.colour_ids and any(n.startswith(color+'_') for n in task['terminal_goals']) for j,color in enumerate(['red','blue','pink'])):return 0.
  return float(goal_score({k:v.mean(0) for k,v in self.current_probs.items()},task['terminal_goals'],self.calibration))
 @torch.no_grad()
 def plan(self,task,dfa,progress,ignore_ltl=False):
  with self.active():return self.searcher.plan(moved(self.state,'cuda'),task,dfa,progress,ignore_ltl)
 def save_state(self,path):
  torch.save({'h':self.state.h,'z':self.state.z,'last_prior':None if self.last_prior is None else {'h':self.last_prior.h,'z':self.last_prior.z},'checkpoint':str(self.checkpoint),'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()},path)
