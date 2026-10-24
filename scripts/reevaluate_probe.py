from pathlib import Path
import json,fcntl,torch
from wm_symbolic.world_model import WorldModel
from wm_symbolic.dataset import Episodes
from wm_symbolic.evaluation import evaluate_data
P=Path('/home/robot/research/projects/wm_symbolic');out=P/'runs/wm_probe_seed0'
lock=(P/'runs/gpu.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
ck=torch.load(out/'checkpoint_01000.pt',map_location='cpu',weights_only=False);m=WorldModel().cuda();m.load_state_dict(ck['model']);d=Episodes(P,'train',128,ck['normalization'])
r=evaluate_data(m,d);r.update(checkpoint='checkpoint_01000.pt',training_unchanged=True,scope='128-transition engineering training probe; not held-out effectiveness')
(out/'metrics_sampling8_01000.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k!='counterexamples'}),flush=True)
