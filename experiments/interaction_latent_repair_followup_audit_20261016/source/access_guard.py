"""Future-only training I/O policy. Not installed in the sealed experiment.
Reject private/test artifacts through real paths; permit explicit training files.
This Python audit hook is defence in depth, not a native-code/OS sandbox.
"""
from pathlib import Path
import os,sys
class TrainingAccessPolicy:
 def __init__(self,project_root,run_root,arm,seed):
  self.project=Path(project_root).resolve();self.run=Path(run_root).resolve();assert self.run.is_relative_to(self.project/'experiments');self.arm=arm;self.seed=seed
  self.protected=[self.project/'data',self.project/'experiments',self.project/'runs']
  names=['inputs.npz','old_inputs.npz','normalization.npz','pairs.json','states.json','chains.json','roots.json']
  self.reads={self.run/'data/public'/n for n in names}
  self.reads|={self.run/'configs'/n for n in ['protocol.json','training_frozen.json','D2_roots.json','D2_root_exclusion.json','schedule_seed0.npz','schedule_seed1.npz']}
  self.reads|={self.run/'models'/f'{p}_seed{s}.pt'for p in ['initial','query']for s in [0,1]}
  # Frozen source integrity verification may read each saved .py, never datasets.
  self.reads|={p.resolve()for p in (self.run/'source').glob('*.py')}
  self.outputs={self.run/'results'/f'gradient_preflight_{arm}_{seed}.json',self.run/'results'/f'train_{arm}_{seed}.json'}
  self.model_output=self.run/'models'/f'{arm}_seed{seed}'
 def check(self,path,mode='r',flags=0):
  if isinstance(path,int):raise PermissionError('Training guard refuses raw fd opens; use explicit authorized paths')
  if not isinstance(path,(str,bytes,os.PathLike)):raise PermissionError('Unrecognized path type')
  p=Path(os.fsdecode(path)).resolve(strict=False);write=(isinstance(mode,str)and any(c in mode for c in'wax+'))or bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
  lexical=Path(os.path.abspath(os.fsdecode(path)));protected=any(p.is_relative_to(root)or lexical.is_relative_to(root)for root in self.protected)
  if not protected:return True
  if '_private'in p.parts or 'candidate'in p.parts:raise PermissionError('Private/evaluator artifacts are forbidden')
  allowed_output=p in self.outputs or(p.parent==self.model_output and(p.name=='selected.json'or(p.name.startswith('update')and p.suffix=='.pt')))
  if write:
   if not allowed_output:raise PermissionError('Write outside declared training outputs')
  elif p not in self.reads and not allowed_output:raise PermissionError('Read outside explicit AB/C training allowlist')
  return True
def install_training_guard(project_root,run_root,arm,seed,access_log):
 policy=TrainingAccessPolicy(project_root,run_root,arm,seed)
 def hook(event,args):
  if event!='open':return
  path=args[0];mode=args[1]if len(args)>1 else'r';flags=args[2]if len(args)>2 and isinstance(args[2],int)else 0
  policy.check(path,mode,flags)
  if isinstance(path,(str,bytes,os.PathLike)):
   p=Path(os.fsdecode(path)).resolve(strict=False)
   if p.is_relative_to(policy.project):access_log.append(str(p))
 sys.addaudithook(hook);return policy
