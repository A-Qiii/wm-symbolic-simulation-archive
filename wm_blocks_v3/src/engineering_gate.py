from paths import GPU_LOCK
import sys,fcntl,gzip,json,hashlib,time,os
import numpy as np
from PIL import Image
from block_env import B,P,Scene,save,serial
from physics_monitor import Monitor
OUT=B/'engineering'/os.environ.get('GATE_REV','gate_01')
PICK=lambda n:['Pick',n,None]
PLACE=lambda n,t:['Place',n,t]
PROGRAMS={
 'stacked_forward':(True,0,[PICK('large'),PLACE('large','B')]),
 'beside_forward':(False,0,[PICK('large'),PLACE('large','B')]),
 'remove_then_large':(True,0,[PICK('small'),PLACE('small','A'),PICK('large'),PLACE('large','B')]),
 'load_then_large':(False,0,[PICK('small'),PLACE('small','large_top'),PICK('large'),PLACE('large','B')]),
 'stacked_reverse':(True,1,[PICK('large'),PLACE('large','A')]),
 'beside_reverse':(False,1,[PICK('large'),PLACE('large','A')]),
 'held_pair_large_A':(False,0,[PICK('small'),PLACE('small','B')]),
 'held_pair_large_B':(False,1,[PICK('small'),PLACE('small','B')]),
 'empty_roundtrip':(False,1,[PICK('large'),PLACE('large','A'),PICK('large'),PLACE('large','B')]),
 'loaded_roundtrip':(True,0,[PICK('large'),PLACE('large','B'),PICK('large'),PLACE('large','A')])}

def layout(i,stacked,reg,name):
 rng=np.random.default_rng(230000000+i);dx=float(rng.uniform(-.008,.008));dy=float(rng.uniform(-.004,.004));g={'seed':230000000+i,'group':f'engineering_{i}','large_xy':[-.08+dx if reg==0 else .18+dx,-.14+dy],'yaw':float(rng.uniform(-.06,.06)),'stacked':stacked,'small_relative_xy':[float(rng.uniform(.027,.033)),float(rng.uniform(-.002,.002))]}
 if name.startswith('held_pair'):g['small_xy']=[.32+dx,-.14+dy]
 return g

def main():
 OUT.mkdir(exist_ok=True);expected=60;rows=[]
 save(OUT/'plan.json',{'engineering_seeds':[230000000+i for i in range(3)],'programs':PROGRAMS,'fresh_repeats':2,'trajectories':expected,'gate':'All 60 complete; opposing bilateral target finger contact before constraint; robot/non-target contacts zero; every physics substep penetration <=2mm; intended carry/no-carry and small-pick large-height <=3mm. No model inference involved.'})
 for i in range(3):
  for name,(stacked,reg,program)in PROGRAMS.items():
   for rep in range(2):
    folder=OUT/f'g{i}_{name}_r{rep}'
    if (folder/'result.json').exists():rows.append(json.load(open(folder/'result.json')));continue
    folder.mkdir(exist_ok=True);g=layout(i,stacked,reg,name);s=Scene(g,capture=(i==0 and rep==0));mon=Monitor(s)
    try:
     actions=[];states=[s.root];boundary_paths=[s.window(folder/'public/b0')]
     if s.valid:
      for action in program:
       x=s.execute(action);actions.append(x);states.append(x['state']);boundary_paths.append(s.window(folder/f'public/b{len(actions)}'))
       if x['status']!='completed':break
     complete=s.valid and len(actions)==len(program)and all(x['status']=='completed'for x in actions);final=states[-1]['objects'];expected_regions=None
     if name in ['stacked_forward','load_then_large']:expected_regions=[1,1]
     if name in ['beside_forward','remove_then_large']:expected_regions=[1,0]
     if name=='stacked_reverse':expected_regions=[0,0]
     if name=='beside_reverse':expected_regions=[0,1]
     if name.startswith('held_pair'):expected_regions=[reg,1]
     if name=='empty_roundtrip':expected_regions=[1,1]
     if name=='loaded_roundtrip':expected_regions=[0,0]
     actual=[final[n]['region']for n in ['large','small']];goal=actual==expected_regions and not any(v['held']for v in final.values());row={'case':folder.name,'layout':g,'valid_initialization':s.valid,'complete':complete,'actions':actions,'expected_regions':expected_regions,'final_regions':actual,'goal':goal,'physics_peak_penetration_m':mon.peak,'unintended_contact_count':len(mon.unintended),'all_attachments_contact_validated':all(a['valid']for a in s.grasp_audits),'grasp_audits':s.grasp_audits,'pass':bool(complete and goal and mon.peak<=.002 and not mon.unintended and all(a['valid']for a in s.grasp_audits)),'boundary_paths':boundary_paths,'phases':s.phases}
     save(folder/'result.json',row)
     with gzip.open(folder/'physics.json.gz','wt')as f:json.dump(mon.samples,f,default=serial)
     if s.capture:
      (folder/'frames').mkdir(exist_ok=True);meta=[]
      for j,fr in enumerate(s.frames):
       for cam in ['rgb_static','rgb_gripper']:Image.fromarray(fr[cam]).save(folder/'frames'/f'{j:04d}_{cam}.png')
       meta.append({k:v for k,v in fr.items()if not k.startswith('rgb_')})
      save(folder/'frame_metadata.json',meta)
     rows.append(row);save(OUT/'progress.json',{'completed':len(rows),'passed':sum(r['pass']for r in rows),'expected':expected,'latest':folder.name});print('GATE',len(rows),folder.name,row['pass'],[(a['status'],a['reason'])for a in actions],mon.peak,flush=True)
    finally:mon.close();s.close()
 save(OUT/'summary.json',{'complete':len(rows)==expected,'passed':sum(r['pass']for r in rows),'denominator':expected,'all_passed':len(rows)==expected and all(r['pass']for r in rows),'cases':[{k:r[k]for k in ['case','pass','final_regions','physics_peak_penetration_m','unintended_contact_count']}for r in rows]})

if __name__=='__main__':
 lock=GPU_LOCK.open('r');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:main()
 finally:fcntl.flock(lock,fcntl.LOCK_UN)
