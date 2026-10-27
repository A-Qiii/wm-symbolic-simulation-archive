"""Local simulator RPC. Private physical truth never returned as model input."""
import sys,json,gzip
from pathlib import Path
import numpy as np,pybullet as p
from PIL import Image
from block_env import Scene,B,save,serial
from physics_monitor import Monitor
from audit_data import support
s=mon=None;folder=None;actions=[];windows=[];spec=None
def close():
 global s,mon
 if mon:mon.close();mon=None
 if s:s.close();s=None
def handle(q):
 global s,mon,folder,actions,windows,spec
 if q['command']=='start':
  close();spec=q['spec'];folder=Path(q['folder']);folder.mkdir(parents=True,exist_ok=False);s=Scene(spec['layout'],capture=q.get('capture',False));mon=Monitor(s,keep_samples=False);actions=[];windows=[s.window(folder/'public/b0')]
  if s.capture:
   original_record=s.record
   def highres():
    frame=s.frames[-1]
    for cam in s.ad.env.cameras:
     static=cam.name=='static';n=600 if static else 256;view=cam.viewMatrix if static else cam.view_matrix;projection=cam.projectionMatrix if static else cam.projection_matrix;pixels=p.getCameraImage(n,n,viewMatrix=view,projectionMatrix=projection,physicsClientId=s.cid);frame['rgb_static_hd'if static else'rgb_gripper_hd']=np.asarray(pixels[2],dtype=np.uint8).reshape(n,n,4)[...,:3].copy()
   def record(*a,**kw):original_record(*a,**kw);highres()
   s.record=record;highres()
  initial_valid=s.valid and support(s.root,'small','large')==bool(spec['layout']['stacked']);spec['initialization_contract_valid']=bool(initial_valid);save(folder/'initial_private.json',{'layout':spec['layout'],'root':s.root,'valid':initial_valid});return {'initial_valid':initial_valid,'public_paths':windows[-1],'control_count':0}
 if q['command']=='execute':
  r=s.execute(q['action']);actions.append(r);windows.append(s.window(folder/f'public/b{len(actions)}'));save(folder/f'action{len(actions)}_private.json',r);return {'status':r['status'],'reason':r['reason'],'public_paths':windows[-1],'control_count':len(s.trace)}
 if q['command']=='finish':
  state=s.private();obj=state['objects'];goal=spec['goal'];released=not any(x['held']for x in obj.values())and s.constraint is None;large=obj['large']['region']==goal['large'];small=obj['small']['region']==goal['small'];always=goal.get('always_small');kept=always is None or(s.root['objects']['small']['region']==always and all(x['regions']['small']==always for x in mon.region_changes));valid=s.valid and spec['initialization_contract_valid']and mon.peak<=.002 and not mon.unintended and all(a['status']=='completed'for a in actions);physical_goal=large and small and released and kept;success=valid and physical_goal and q['stop']=='goal_reached';result={'success':bool(success),'physical_goal':bool(physical_goal),'stop':q['stop'],'physics_valid':bool(valid),'initialization_valid':spec['initialization_contract_valid'],'maximum_penetration_m':mon.peak,'unintended_contacts':mon.unintended,'small_constraint_kept':bool(kept),'actions':[x['action']for x in actions],'final_state':state,'physics_steps':mon.n,'region_changes':mon.region_changes,'goal':goal};save(folder/'score_private.json',result);save(folder/'grasp_audits_private.json',s.grasp_audits)
  with gzip.open(folder/'control_trace_private.json.gz','wt')as f:json.dump(s.trace,f,default=serial)
  if s.capture:
   (folder/'frames').mkdir(exist_ok=True);metadata=[]
   for i,fr in enumerate(s.frames):
    for cam in ['rgb_static','rgb_gripper','rgb_static_hd','rgb_gripper_hd']:
     if cam in fr:Image.fromarray(fr[cam]).save(folder/'frames'/f'{i:04d}_{cam}.png')
    metadata.append({k:v for k,v in fr.items()if not k.startswith('rgb_')})
   save(folder/'frame_metadata.json',metadata)
  return {'score_saved':True}
 if q['command']=='close':close();return {'closed':True}
 raise ValueError(q['command'])
def main():
 try:
  for line in sys.stdin:
   try:print('BLOCK_RPC '+json.dumps({'ok':True,**handle(json.loads(line))}),flush=True)
   except Exception as e:print('BLOCK_RPC '+json.dumps({'ok':False,'error':repr(e)}),flush=True);raise
 finally:close()

if __name__=='__main__':main()
