"""Physical collection for frozen six-skill study; private targets separated."""
from pathlib import Path
from collections import deque, Counter
import argparse, json, hashlib, datetime, time, os, fcntl, signal, gzip
import numpy as np
import pybullet as p
import hydra, imageio.v2 as imageio
from omegaconf import OmegaConf
from wm_symbolic.env_adapter import CalvinAdapter, save_npz
from wm_symbolic.physical_oracle import PhysicalOracle
from wm_symbolic.calibrated_labels import calibrated_labels_at
from wm_symbolic.core_ownership import drawer_ownership
from evaluate_physical_oracle import physics_guard

P=Path('/home/robot/research/projects/wm_symbolic'); S=P/'experiments/role_generalization_20261019'; B=S/'data'; R=S/'results'
COLORS=['red','blue','pink']; stop=False
def requested(sig,frame):
 global stop
 stop=True
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,obj):
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False));tmp.replace(path)
def plain(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [plain(v) for v in x]
 return x
def label(env):
 info=env.env.get_info(); result=calibrated_labels_at(info,env.config.scene); oracle=PhysicalOracle(env);held=oracle.held()
 for j,c in enumerate(COLORS):
  name='block_'+c
  if name in held:result['categorical'][j]=6;result['categorical_valid'][j]=True
  elif result['categorical'][j]==6:result['categorical'][j]=7;result['categorical_valid'][j]=False
 result['holding_derived']=[name in held for name in ['block_'+c for c in COLORS]]
 result['hand_empty_derived']=not bool(held) if all(result['categorical_valid'][:3]) else None
 result['held_definition']='bilateral force-bearing descendant finger contacts; no external support'
 return drawer_ownership(plain(result),info)
def hist_save(d,prefix,history):
 frames=list(history);frames=[frames[0]]*(4-len(frames))+frames
 for j,obs in enumerate(frames):save_npz(d/f'{prefix}_{j}.npz',obs)
def run_episode(root,branch,oracle):
 eid=f"{root['root_id']}_b{branch}";d=B/'episodes'/eid
 if d.exists():d.rename(d.with_name(eid+'_interrupted_'+str(int(time.time()))))
 d.mkdir(parents=True);env=CalvinAdapter(P,root['environment'],root['seed'],tabletop=True,segmentation=True)
 try:
  for obj in env.env.scene.movable_objects:
   val=root['layout'][obj.name];obj.initial_pos=np.array(val[:3]);obj.initial_orn=np.array(p.getQuaternionFromEuler(val[3:]))
  obs=env.reset()
  for _ in range(24):obs=env.step([obs['robot_obs'][:3],obs['robot_obs'][3:6],1])
  # Same family tabletop geometry verified before role-dependent motor setup.
  base=env.env.get_state_obs();fd=B/'families'/root['family_id'];fd.mkdir(parents=True,exist_ok=True);refbase=fd/'tabletop_private.npz'
  compare={'family_id':root['family_id'],'root_id':root['root_id'],'branch':branch}
  if not refbase.exists():np.savez_compressed(refbase,**base)
  else:
   with np.load(refbase) as z:
    compare['scene_max_abs']=float(np.max(np.abs(z['scene_obs']-base['scene_obs'])));compare['robot_max_abs']=float(np.max(np.abs(z['robot_obs']-base['robot_obs'])))
   assert max(compare['scene_max_abs'],compare['robot_max_abs'])<=1e-5,compare
  dump(d/'family_base_check.json',compare)
  dump(d/'physical_identity_private.json',{'family_id':root['family_id'],'layout':root['layout'],'objects':{o.name:{'uid':o.uid,'file':str(env.config.scene.objects.movable_objects[o.name].file),'mass':float(p.getDynamicsInfo(o.uid,-1,physicsClientId=env.env.cid)[0])}for o in env.env.scene.movable_objects},'calibration':plain(env.calibration()),'role_intervention':'Only setup-held object and requested target differ; actual body identity, asset, dimensions, mass and base pose stay fixed.'})
  # Setup is physically executed and logged, never counted as requested actions.
  setup=[];setup_controls=[]
  def setup_record(obs,act):setup_controls.append(act)
  ctl=PhysicalOracle(env,setup_record)
  for skill in root['setup_skills']:
   with physics_guard(env):res=ctl.run(skill)
   setup.append({'skill':skill,**res})
  dump(d/'setup_private.json',{'results':setup,'controls':setup_controls,'planned_initial_state':root['initial_state'],'actual_initial_labels':label(env)})
  state=env.env.get_state_obs();np.savez_compressed(d/'root_reset_private.npz',**state)
  rootd=B/'roots'/root['root_id'];rootd.mkdir(parents=True,exist_ok=True)
  heldref=B/'families'/root['family_id']/('held_'+root['held_color']+'_private.npz')
  if not heldref.exists():np.savez_compressed(heldref,**state)
  else:
   with np.load(heldref) as z:assert all(np.max(np.abs(z[k]-state[k]))<=1e-5 for k in ['scene_obs','robot_obs']),'Same family/H setup changed across target roles'

  check={'branch':branch,'setup_all_successful':all(x['status']=='success' for x in setup)}
  ref=rootd/'branch1_initial_private.npz'
  if branch==1:np.savez_compressed(ref,**state)
  else:
   with np.load(ref) as z:
    check['scene_max_abs_difference']=float(np.max(np.abs(z['scene_obs']-state['scene_obs'])))
    check['robot_max_abs_difference']=float(np.max(np.abs(z['robot_obs']-state['robot_obs'])))
   if max(check['scene_max_abs_difference'],check['robot_max_abs_difference'])>1e-5:raise RuntimeError('Branches do not share the same physical root: '+str(check))
  dump(d/'root_replay_check.json',check)
  history=deque(list(env.history),maxlen=4);hist_save(d,'root_observation',history)
  rows=[]
  for step,skill in enumerate(root['branches'][branch]):
   s=d/f'{step:02d}_{skill}';s.mkdir();initial=env.observe();startinfo=env.env.get_info();before=label(env);startstate=env.env.get_state_obs();save_npz(s/'initial_public.npz',initial)
   frames={cam:imageio.get_writer(str(s/(cam+'.mp4')),fps=30,macro_block_size=1,codec='libx264',quality=7) for cam in initial['rgb_obs']}
   for cam,w in frames.items():w.append_data(initial['rgb_obs'][cam])
   actions=[];trace=[];startpos=np.array([startinfo['scene_info']['movable_objects']['block_'+c]['current_pos'] for c in COLORS]);maxdisp=np.zeros(3);first=[None]*3
   def record(obs,act):
    actions.append({**act,**applied_commands[-1]});history.append(obs)
    for cam,w in frames.items():w.append_data(obs['rgb_obs'][cam])
    positions=np.array([env.env.scene.get_info()['movable_objects']['block_'+c]['current_pos'] for c in COLORS]);dist=np.linalg.norm(positions-startpos,axis=1);maxdisp[:]=np.maximum(maxdisp,dist)
    for j in range(3):
     if first[j] is None and dist[j]>.02:first[j]=len(actions)
    contacts=[list(c) for c in p.getContactPoints(physicsClientId=env.env.cid) if c[9]>.01]
    trace.append({'robot_obs':obs['robot_obs'].tolist(),'scene_obs':env.env.scene.get_obs().tolist(),'objects':{o.name:o.get_state().tolist() for o in env.env.scene.movable_objects},'contacts':contacts})
   applied_commands=[];original_step=env.step;fault_active=branch==2 and step==1
   def execution_step(action):
    requested=int(action[2]);applied=1 if fault_active and requested==-1 else requested
    output=original_step([np.asarray(action[0]),np.asarray(action[1]),applied])
    applied_commands.append({'requested_xyz':np.asarray(action[0]).tolist(),'requested_quaternion':np.asarray(action[1]).tolist(),'requested_gripper':requested,'applied_gripper':applied,'declared_execution_disturbance':bool(fault_active)})
    return output
   env.step=execution_step
   ctl=PhysicalOracle(env,record)
   try:
    with physics_guard(env):ctl.execute(skill)
    result=ctl.last_audit_private
   finally:
    env.step=original_step
    for w in frames.values():w.close()
   final=env.observe();endstate=env.env.get_state_obs();after=label(env);save_npz(s/'final_public.npz',final)
   if not actions:
    assert np.array_equal(startstate['scene_obs'],endstate['scene_obs']) and np.array_equal(startstate['robot_obs'],endstate['robot_obs'])
    assert before==after,'Zero-action labels changed'
   hist_save(s,'boundary_history',history)
   official=skill in oracle.get_task_info_for_set(startinfo,env.env.get_info(),{skill})
   goal_success=bool(official and result['status']=='success')
   continuous=np.stack([startstate['scene_obs']]+[np.array(x['scene_obs']) for x in trace]);side=float(np.max(np.abs(continuous[:,[0,2,3,4,5]]-continuous[0,[0,2,3,4,5]])))
   result.update(declared_execution_disturbance='gripper_open_override'if fault_active else None,requested_applied_gripper_changes=sum(a['requested_gripper']!=a['applied_gripper']for a in applied_commands),skill=skill,official_success=official,goal_success=goal_success,zero_action_state_identical=not actions,continuous_unrequested_device_max_m=side,strict_device_side_effect=side>.003,unreviewed_incidental_motion=side>.003,control_samples_including_start=len(actions)+1,physics_guard_passed=True,controller_sha256=sha(P/'src/wm_symbolic/physical_oracle.py'))
   labels={'schema':'core_labels_v1','before':before,'after':after,'success':goal_success,'events':(maxdisp>.02).astype(int).tolist(),'events_valid':[True]*3,'event_max_displacement_m':maxdisp.tolist(),'event_first_control_step':first,'role':'offline_targets_only','control_samples_including_start':len(actions)+1}
   dump(s/'labels_private.json',labels);dump(s/'result_private.json',result);dump(s/'controls.json',actions)
   with gzip.open(s/'trace_private.json.gz','wt') as f:json.dump(trace,f)
   dump(s/'endpoint_info_private.json',plain({'before':startinfo,'after':env.env.get_info()}))
   row={'episode_id':eid,'root_id':root['root_id'],'family_id':root['family_id'],'split':root['split'],'dev_role':root['dev_role'],'environment':root['environment'],'step':step,'skill':skill,'executor':{'checkpoint':PhysicalOracle.VERSION,'budget_steps':900},'obs_before':str(s/'initial_public.npz'),'obs_after':str(s/'final_public.npz'),'labels_path':str(s/'labels_private.json'),'low_level_log':str(s/'controls.json'),'collection_style':'matched_execution_disturbance','branch_index':branch,'branch_role':['nominal_primary','nominal_matched','disturbed_matched'][branch],'controller_sha256':result['controller_sha256']}
   rows.append(row);dump(d/'committed_rows.json',rows)
   print(json.dumps({'episode':eid,'step':step,'controls':len(actions),'status':result['status'] if root['split']!='evaluation' else 'sealed'}),flush=True)
  dump(d/'COMPLETE.json',{'rows':len(rows),'root_id':root['root_id'],'controller_unchanged':True});return rows
 finally:env.close()