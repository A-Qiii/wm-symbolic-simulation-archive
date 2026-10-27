"""128 real training transitions: input isolation, finite loss, scratch optimization."""
from paths import GPU_LOCK
from paths import ROOT as B, RUNTIME as P
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import json,fcntl,hashlib
import numpy as np,torch
from pathlib import Path
from models import RGBState
from train import state_loss,tensor,subset,state_score
def main():
 plan=json.load(open(B/'configs/experiment_plan.json'));meta=json.load(open(B/'cache/v1/fit/state_labels.json'));feat=np.load(B/'cache/v1/fit/all_features.npz')['features'];allowed=set(plan['experiments']['state']['arms']['holdout']['train']);episodes=[];transitions=0
 for e in plan['experiments']['state']['episodes']:
  if e['id']not in allowed:continue
  episodes.append(e['id']);transitions+=len(e['actions'])
  if transitions>=128:break
 rows=[i for i,r in enumerate(meta)if r['episode_id']in episodes];labels=[meta[i]for i in rows];raw=feat[rows];mu=raw.reshape(-1,raw.shape[-1]).mean(0);sd=np.maximum(raw.reshape(-1,raw.shape[-1]).std(0),.001);x=(raw-mu)/sd
 for eid in episodes:
  e=json.load(open(B/'data/v1/state'/eid/'episode.json'));assert e['physical_valid']and not e['forbidden_state_seen']
  for paths in e['public_windows']:
   for path in paths:
    with np.load(path)as z:assert set(z.files)=={'rgb_static','rgb_gripper','robot_obs'}
 data={'features':tensor(x),'xyz':tensor([r['xyz']for r in labels]),'shape':tensor([r['shape']for r in labels]),'region':tensor([r['region']for r in labels],True),'held':tensor([r['held']for r in labels])};torch.manual_seed(41900);m=RGBState(x.shape[-1]).cuda();opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4);before=float(state_loss(m,data));rng=np.random.default_rng(41900)
 for _ in range(200):
  ix=rng.integers(len(rows),size=64);opt.zero_grad(set_to_none=True);loss=state_loss(m,subset(data,ix));assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
 after=float(state_loss(m,data));assert after<before;out={'transitions':transitions,'episodes':len(episodes),'states':len(rows),'training_input_keys_exact':True,'state_holdout_absent_from_all_selected_histories':True,'initial_loss':before,'final_loss':after,'scratch_updates':200,'scratch_weights_discarded':True,'no_development_or_test_used':True,'fit_on_training_only':state_score(m,data)};(B/'runs/v1').mkdir(parents=True,exist_ok=True);json.dump(out,open(B/'runs/v1/smoke128.json','w'),indent=2);print('SMOKE128',out,flush=True)
if __name__=='__main__':
 lock=GPU_LOCK.open('r');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4)
 try:main()
 finally:fcntl.flock(lock,fcntl.LOCK_UN)
