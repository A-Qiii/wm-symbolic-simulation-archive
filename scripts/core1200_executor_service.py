"""Isolated physical process. IPC returns only past/current public frames."""
from pathlib import Path
from collections import deque
import argparse,json,time,traceback,gzip
import numpy as np
import pybullet as p
import imageio.v2 as imageio
import hydra
from omegaconf import OmegaConf
from wm_symbolic.env_adapter import CalvinAdapter,save_npz
from wm_symbolic.physical_oracle import PhysicalOracle
from wm_symbolic.gpu_phase import gpu_phase
from core1200_collect import label,dump,physics_guard,COLORS
P=Path('/home/robot/research/projects/wm_symbolic');B=P/'data/core1200_v1'
def main():
 a=argparse.ArgumentParser();a.add_argument('--root',required=True);a.add_argument('--run',type=Path,required=True);args=a.parse_args();out=args.run
 for name in ['requests','responses','public','evaluator_only']:(out/name).mkdir(exist_ok=True)
 root=next(r for r in json.loads((B/'roots_manifest.json').read_text()) if r['root_id']==Path(args.root).name);env=None
 try:
  with gpu_phase(P,'core_frozen_executor_initialize'):
   env=CalvinAdapter(P,root['environment'],root['seed'],tabletop=True,segmentation=True)
   for obj in env.env.scene.movable_objects:
    v=root['layout'][obj.name];obj.initial_pos=np.array(v[:3]);obj.initial_orn=np.array(p.getQuaternionFromEuler(v[3:]))
   obs=env.reset()
   for _ in range(24):obs=env.step([obs['robot_obs'][:3],obs['robot_obs'][3:6],1])
   setup=[];ctl=PhysicalOracle(env)
   for skill in root['setup_skills']:
    with physics_guard(env):setup.append({'skill':skill,**ctl.run(skill)})
   history=deque(env.history,maxlen=4);dump(out/'evaluator_only/root.json',{'root':root,'setup':setup,'label':label(env)})
   actual=env.env.get_state_obs();np.savez_compressed(out/'evaluator_only/root_state.npz',**actual)
   # Compare with the independent data root when it exists, no task-driven reset.
   with np.load(B/'roots'/root['root_id']/'branch0_initial_private.npz') as z:
    error=max(float(np.max(np.abs(z[k]-actual[k]))) for k in ['robot_obs','scene_obs'])
   assert error<1e-5,'Closed-loop root differs from frozen test data root'
  def publish(index):
   d=out/'public'/index;d.mkdir();hist=list(history);hist=[hist[0]]*(4-len(hist))+hist;paths=[]
   for i,obs in enumerate(hist):
    path=d/f'{i}.npz';save_npz(path,obs);paths.append(str(path))
   return paths
  dump(out/'ready.json',{'public_frames':publish('root'),'hidden_feedback_returned':False})
  oracle=hydra.utils.instantiate(OmegaConf.load(P/'vendor/calvin/calvin_models/conf/callbacks/rollout/tasks/new_playtable_tasks.yaml'));index=0
  while True:
   req=out/'requests'/f'{index:03d}.json'
   if not req.exists():time.sleep(.02);continue
   command=json.loads(req.read_text())
   if command['command']=='shutdown':dump(out/'responses'/req.name,{'closed':True});break
   assert command['command'] in ['skill','reobserve']
   skill=command.get('skill','reobserve');d=out/'evaluator_only'/f'{index:03d}_{skill}';d.mkdir();acts=[];trace=[];before=label(env);before_info=env.env.get_info();obs=env.observe();startpos=np.array([env.env.scene.get_info()['movable_objects']['block_'+c]['current_pos'] for c in COLORS]);maxdisp=np.zeros(3)
   writers={cam:imageio.get_writer(str(d/f'{cam}.mp4'),fps=30,macro_block_size=1,quality=7) for cam in obs['rgb_obs']}
   for cam,w in writers.items():w.append_data(obs['rgb_obs'][cam])
   def record(obs,act):
    acts.append(act);history.append(obs)
    for cam,w in writers.items():w.append_data(obs['rgb_obs'][cam])
    pos=np.array([env.env.scene.get_info()['movable_objects']['block_'+c]['current_pos'] for c in COLORS]);maxdisp[:]=np.maximum(maxdisp,np.linalg.norm(pos-startpos,axis=1))
    trace.append({'robot_obs':obs['robot_obs'].tolist(),'scene_obs':env.env.scene.get_obs().tolist(),'objects':{o.name:o.get_state().tolist() for o in env.env.scene.movable_objects},'contacts':[list(c) for c in p.getContactPoints(physicsClientId=env.env.cid) if c[9]>.01]})
   ctl=PhysicalOracle(env,record)
   try:
    with gpu_phase(P,'core_frozen_executor_control'),physics_guard(env):
     if command['command']=='skill':ctl.execute(skill);private=ctl.last_audit_private
     else:ctl.hold(4,int(np.sign(obs['robot_obs'][-1]) or 1));private={'status':'reobserved','controls':4}
   finally:
    for w in writers.values():w.close()
   official=command['command']=='skill' and skill in oracle.get_task_info_for_set(before_info,env.env.get_info(),{skill})
   dump(d/'summary.json',{'before':before,'after':label(env),'events':(maxdisp>.02).astype(int).tolist(),'event_max_displacement_m':maxdisp.tolist(),'result':private,'official_success':official,'control_frames_including_start':len(acts)+1});dump(d/'controls.json',acts)
   with gzip.open(d/'trace_private.json.gz','wt') as f:json.dump(trace,f)
   dump(out/'responses'/req.name,{'public_frames':publish(f'{index:03d}'),'control_steps':len(acts),'hidden_feedback_returned':False});index+=1
 except Exception:
  dump(out/'error.json',{'error':traceback.format_exc()});raise
 finally:
  if env is not None:env.close()
if __name__=='__main__':main()
