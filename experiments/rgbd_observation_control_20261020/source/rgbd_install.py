from pathlib import Path
import shutil,json,hashlib
P=Path('/home/robot/research/projects/wm_symbolic');O=P/'experiments/object_recurrent_state_run_20261019';S=P/'experiments/rgbd_observation_control_20261020';W=Path('/home/robot/Documents/Codex/2026-10-13/task-4')
shutil.copyfile(W/'rgbd_data.py',S/'source/data.py')
with open(S/'source/models.py','a')as f:f.write('''
class ObjectAug(ObjectGRU):
 def __init__(self,interface=None):
  super().__init__(interface);self.object_geometry=torch.nn.Linear(16,512,bias=False);self.context_geometry=torch.nn.Linear(32,512,bias=False)
  torch.nn.init.zeros_(self.object_geometry.weight);torch.nn.init.zeros_(self.context_geometry.weight)
 def observe(self,prior,objects,context,availability):
  objects=objects[...,:512]+self.object_geometry(objects[...,512:]);context=context[...,:512]+self.context_geometry(context[...,512:])
  return super().observe(prior,objects,context,availability)
''')
t=(O/'source/train.py').read_text().replace('from models import Interface,ObjectGRU,GlobalGRU','from models import Interface,ObjectGRU,GlobalGRU,ObjectAug').replace("return {'global':GlobalGRU,'object':ObjectGRU}[arm](interface)","return ObjectAug(interface)").replace("data=load('train',seed)","data=load('train',seed,arm=arm)")
(S/'source/train.py').write_text(t)
e=(O/'source/evaluate.py').read_text().replace('data=load(split,seed)','data=load(split,seed,arm=arm)').replace("for split in ['train','development','historical_evaluation']:assess(arm,seed,split)","for split in ['train','development','historical_evaluation','diagnosed_confirmation']:assess(arm,seed,split)")
(S/'source/evaluate.py').write_text(e)
for n in ['rgbd_setup.py','rgbd_install.py']:shutil.copyfile(W/n,S/'source'/n)
print('installed sources')
