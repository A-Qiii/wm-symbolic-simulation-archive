"""Independent confirmation references, unlocked only after development gate."""
from paths import GPU_LOCK
from paths import ROOT as B, RUNTIME as P
import json,sys,fcntl
from pathlib import Path
OUT=B/'data/v3/confirmation'
def read(p):return json.load(open(p))
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def manifest():
 from protocol import COMMON,POSITIVE,layout
 plan=read(B/'configs/v3_plan.json');recipes={**COMMON,**POSITIVE['transition'],'A_beside_remove_then_move':{**COMMON['A_remove_then_move'],'stacked':False}};episodes=[]
 for g in plan['confirmation_groups']:
  for name in plan['confirmation_recipes']:
   r=recipes[name];episodes.append({'id':g['id']+'__'+name,'group':g['id'],'split':'test','recipe':name,'layout':layout(g,r),'actions':r['actions']})
 x={'groups':plan['confirmation_groups'],'episodes':episodes,'locked_before_collection':True};p=B/'configs/v3_confirmation_manifest.json'
 if p.exists():assert read(p)==x
 else:save(p,x)
 return x
def collect():
 from block_env import Scene
 from physics_monitor import Monitor
 assert read(B/'configs/v3_models_frozen.json')['development_gate_passed'];mfest=manifest()
 for n in ['block_env.py','kinematic_guard.py','physics_monitor.py']:assert(B/'src'/n).read_bytes()==(B/'data/v1/frozen_source'/n).read_bytes()
 failures=[]
 for i,e in enumerate(mfest['episodes']):
  folder=OUT/e['id'];done=folder/'episode.json'
  if done.exists():
   if not read(done)['physical_valid']:failures.append(e['id'])
   continue
  if folder.exists():raise RuntimeError('Incomplete confirmation retained: '+str(folder))
  folder.mkdir(parents=True);s=Scene(e['layout']);mon=Monitor(s,keep_samples=False)
  try:
   states=[s.private()];windows=[s.window(folder/'public/b0')];actions=[];valid=s.valid
   if valid:
    for a in e['actions']:
     r=s.execute(a);actions.append(r);states.append(s.private());windows.append(s.window(folder/f'public/b{len(actions)}'))
     if r['status']!='completed':valid=False;break
   valid=valid and mon.peak<=.002 and not mon.unintended
   from audit_data import support
   intended=bool(e['layout']['stacked']);stratum=support(states[0],'small','large')==intended;valid=valid and stratum
   save(done,{**e,'layout_private':e['layout'],'states':states,'public_windows':windows,'actions':actions,'physical_valid':bool(valid),'initialization_valid':bool(s.valid and stratum),'maximum_penetration_m':mon.peak,'physics_region_changes':mon.region_changes})
   if not valid:failures.append(e['id'])
  finally:mon.close();s.close()
  save(B/'STATUS.json',{'version':'v3','stage':'confirmation_collection','episodes':i+1,'planned':len(mfest['episodes']),'physical_failures':len(failures)});print('V2_CONFIRM_DATA',i+1,len(failures),flush=True)
 save(OUT/'complete.json',{'episodes':len(mfest['episodes']),'planned_transitions':sum(len(e['actions'])for e in mfest['episodes']),'failures_retained':failures})
def cache():
 import numpy as np
 from rgb_front import RGBFront
 assert(OUT/'complete.json').exists();out=B/'cache/v3/confirmation';out.mkdir(parents=True,exist_ok=True);front=RGBFront();rows=[];values=[]
 try:
  for e in read(B/'configs/v3_confirmation_manifest.json')['episodes']:
   f=OUT/e['id']/'episode.json';r=read(f)
   for i,paths in enumerate(r['public_windows']):
    dest=out/(e['id']+f'__b{i}.npz')
    if dest.exists():v=np.load(dest)['features']
    else:v,_=front.extract(paths);np.savez_compressed(dest,features=v)
    rows.append({'episode':e['id'],'group':e['group'],'boundary':i,'private_episode':str(f)});values.append(v)
  np.savez_compressed(out/'all_features.npz',features=np.stack(values));save(out/'state_rows.json',rows);save(out/'complete.json',{'states':len(rows),'test_fitting':False})
 finally:front.close()
if __name__=='__main__':
 lock=GPU_LOCK.open('r');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:collect()if sys.argv[1]=='collect'else cache()
 finally:fcntl.flock(lock,fcntl.LOCK_UN)
