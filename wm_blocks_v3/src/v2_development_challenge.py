"""Development-only lawful cross-program reference, never training/test.
Same v1 physics and original development layouts; no held-out reverse transfer.
"""
from paths import GPU_LOCK
from paths import ROOT as B, RUNTIME as P
import os,json,sys,hashlib,fcntl
from pathlib import Path
OUT=B/'data/v2/development_cross'
def read(p):return json.load(open(p))
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def collect():
 from block_env import Scene
 from physics_monitor import Monitor
 plan=read(B/'configs/experiment_plan.json');episodes=[e for e in plan['experiments']['transition']['episodes']if e['split']=='development'and e['recipe']=='A_load_then_move'];assert len(episodes)==20
 for name in ['block_env.py','kinematic_guard.py','physics_monitor.py']:
  assert(B/'src'/name).read_bytes()==(B/'data/v1/frozen_source'/name).read_bytes()
 for i,e in enumerate(episodes):
  folder=OUT/e['group'];done=folder/'episode.json'
  if done.exists():continue
  if folder.exists():raise RuntimeError('Incomplete development run retained: '+str(folder))
  folder.mkdir(parents=True);s=Scene(e['layout']);m=Monitor(s,keep_samples=False)
  try:
   assert s.valid;states=[s.private()];windows=[s.window(folder/'public/b0')];actions=[]
   for a in [['Pick','small',None],['Place','small','A'],['Pick','large',None],['Place','large','B']]:
    r=s.execute(a);actions.append(r);states.append(s.private());windows.append(s.window(folder/f'public/b{len(actions)}'));assert r['status']=='completed',r['reason']
   assert not m.unintended and m.peak<=.002
   save(done,{'id':e['group']+'__A_beside_remove_then_move','group':e['group'],'split':'development','purpose':'Cross-program diagnosis and development selection only; never fitted','source_v1_episode':e['id'],'layout_private':e['layout'],'actions':actions,'states':states,'public_windows':windows,'physical_valid':True,'maximum_penetration_m':m.peak})
  finally:m.close();s.close()
  save(B/'STATUS.json',{'version':'v2','stage':'development_cross_diagnosis','completed':i+1,'total':20});print('DEV_CROSS',i+1,flush=True)
 save(OUT/'complete.json',{'episodes':20,'transitions':80,'training':False,'test':False,'original_group_split_preserved':True})
def cache():
 import numpy as np
 from rgb_front import RGBFront
 assert(OUT/'complete.json').exists();front=RGBFront();out=B/'cache/v2/development_cross';out.mkdir(parents=True,exist_ok=True);rows=[];values=[]
 try:
  for f in sorted(OUT.glob('*/episode.json')):
   e=read(f)
   for i,paths in enumerate(e['public_windows']):
    dest=out/(e['id']+f'__b{i}.npz')
    if dest.exists():v=np.load(dest)['features']
    else:v,_=front.extract(paths);np.savez_compressed(dest,features=v)
    rows.append({'episode':e['id'],'group':e['group'],'boundary':i,'private_episode':str(f)});values.append(v)
  np.savez_compressed(out/'all_features.npz',features=np.stack(values));save(out/'state_rows.json',rows);save(out/'complete.json',{'states':len(rows),'training':False,'test':False})
 finally:front.close()
if __name__=='__main__':
 lock=GPU_LOCK.open('r');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:collect()if sys.argv[1]=='collect'else cache()
 finally:fcntl.flock(lock,fcntl.LOCK_UN)
