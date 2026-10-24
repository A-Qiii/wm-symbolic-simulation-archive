from pathlib import Path
import sys,os,json,hashlib,datetime,time,csv,collections,ast,copy,fcntl,signal
import numpy as np
P=Path('/home/robot/research/projects/wm_symbolic');R=P/'experiments/failure_recovery_control_20261019';I=P/'experiments/initial_state_audit_20261019';C=P/'experiments/paired_outcome_training_20261019';U=P/'experiments/observation_state_update_20261019'
sys.path.insert(0,str(R/'source'))
from common import read,save,sha,statehash,csvwrite,rel,truth,correct,now,CELL,COLORS,SKILLS
from dataset import NewEpisodes
from old_dataset import CoreEpisodes
import train as legacy_train
from losses import losses,backward_routed,clip_groups
from wm_symbolic.world_model import WorldModel
import torch
os.environ['MPLCONFIGDIR']=str(U/'cache');os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
def guard(stage):
 def audit(event,args):
  if event=='socket.connect':raise RuntimeError('Offline')
  if event!='open'or not isinstance(args[0],(str,bytes)):return
  p=Path(os.fsdecode(args[0])).resolve();m=args[1];f=args[2]if len(args)>2 else 0
  if not p.is_relative_to(P):return
  writing=(isinstance(m,str)and any(c in m for c in 'wa+'))or(isinstance(f,int)and f&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
  if writing and not p.is_relative_to(U):raise RuntimeError('Historical output write forbidden '+str(p))
  if any(t in str(p)for t in ['externalC','C_core1200','D_core1200','independent_identity_D','independent_objective_D','calvin_scene_C','calvin_scene_D']):raise RuntimeError('C/D forbidden')
  if stage=='train'and('_recovery_evaluation_'in str(p)or p.name.endswith('_evaluation.npz')):raise RuntimeError('Historical exploratory evaluation blocked during training')
 sys.addaudithook(audit)
def prepared():
 norm=read(CELL/'configs/normalization.json');old=CoreEpisodes(P,'train',norm);new=NewEpisodes('train');new.episodes=[e for e in new.episodes if e['branch']in [1,2]];new.transitions=6*len(new.episodes);dev=NewEpisodes('development');return old,new,dev,norm
